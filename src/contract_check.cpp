#include "moe/config.hpp"
#include "moe/protocol.hpp"

#include <cstdlib>
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool parse_world_size(std::vector<std::string>& arguments, int& world_size,
                      std::string& error) {
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    if (arguments[i] != "--inspect-ranks") {
      continue;
    }
    if (i + 1 >= arguments.size()) {
      error = "--inspect-ranks requires a positive integer";
      return false;
    }
    try {
      const auto parsed = std::stol(arguments[i + 1]);
      if (parsed <= 0 || parsed > 1'000'000) {
        error = "--inspect-ranks must be between 1 and 1000000";
        return false;
      }
      world_size = static_cast<int>(parsed);
    } catch (const std::exception&) {
      error = "--inspect-ranks requires a positive integer";
      return false;
    }
    arguments.erase(arguments.begin() + static_cast<std::ptrdiff_t>(i),
                    arguments.begin() + static_cast<std::ptrdiff_t>(i + 2));
    return true;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> arguments;
  for (int i = 1; i < argc; ++i) {
    arguments.emplace_back(argv[i]);
  }

  int inspected_world_size = 4;
  std::string inspection_error;
  if (!parse_world_size(arguments, inspected_world_size, inspection_error)) {
    std::cerr << "Error: " << inspection_error << '\n';
    return EXIT_FAILURE;
  }

  const auto parsed = moe::parse_command_line(arguments);
  if (parsed.help_requested) {
    std::cout << moe::command_line_help()
              << "\nContract-check-only option:\n  --inspect-ranks P\n";
    return EXIT_SUCCESS;
  }
  if (!parsed.ok) {
    std::cerr << "Error: " << parsed.error << "\n\n" << moe::command_line_help();
    return EXIT_FAILURE;
  }

  const auto& config = parsed.config;
  std::cout << "Frozen MoE protocol check\n"
            << "  N=" << config.tokens << ", d=" << config.dimension
            << ", h=" << config.hidden_dimension << ", E=" << config.experts
            << ", top-k=" << config.top_k << '\n'
            << "  parallelism=" << moe::to_string(config.parallelism)
            << ", router-mode=" << moe::to_string(config.router_mode)
            << ", skew=" << moe::to_string(config.skew)
            << ", hot-layout=" << moe::to_string(config.hot_layout) << '\n'
            << "  placement=" << moe::to_string(config.placement)
            << ", communication=" << moe::to_string(config.communication)
            << ", omp-schedule=" << moe::to_string(config.omp_schedule) << '\n'
            << "  seeds: model=" << config.model_seed << ", input=" << config.input_seed
            << ", routing=" << config.routing_seed
            << ", calibration=" << config.calibration_seed << '\n';

  std::uint64_t partitioned_tokens = 0;
  for (int rank = 0; rank < inspected_world_size; ++rank) {
    const auto range = moe::contiguous_token_range(config.tokens, rank, inspected_world_size);
    partitioned_tokens += range.count;
    std::cout << "  rank " << rank << ": tokens [" << range.begin << ", "
              << range.begin + range.count << ")\n";
  }
  if (partitioned_tokens != config.tokens) {
    std::cerr << "Error: contiguous partition did not cover all tokens\n";
    return EXIT_FAILURE;
  }

  for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
    const int owner = moe::round_robin_expert_owner(expert, inspected_world_size);
    if (owner < 0 || owner >= inspected_world_size) {
      std::cerr << "Error: invalid expert owner\n";
      return EXIT_FAILURE;
    }
  }
  std::cout << "RESULT: PASS\n";
  return EXIT_SUCCESS;
}
