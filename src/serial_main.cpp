#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/serial.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool finite_output(const std::vector<float>& output) {
  for (const float value : output) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> arguments;
  for (int i = 1; i < argc; ++i) {
    arguments.emplace_back(argv[i]);
  }
  const auto parsed = moe::parse_command_line(arguments);
  if (parsed.help_requested) {
    std::cout << moe::command_line_help();
    return EXIT_SUCCESS;
  }
  if (!parsed.ok) {
    std::cerr << "Error: " << parsed.error << "\n\n" << moe::command_line_help();
    return EXIT_FAILURE;
  }
  const auto& config = parsed.config;
  if (config.expert_backend != moe::ExpertBackend::kCpu) {
    std::cerr << "Error: the CUDA Expert backend has not been implemented yet\n";
    return EXIT_FAILURE;
  }

  const auto model = moe::generate_model(config);
  const auto tokens = moe::generate_tokens(config, 0, config.tokens);
  const auto trace = config.router_mode == moe::RouterMode::kTrace
                         ? moe::generate_controlled_trace(config)
                         : std::vector<moe::RoutingDecision>{};
  const auto* trace_pointer = config.router_mode == moe::RouterMode::kTrace ? &trace : nullptr;

  const auto start = std::chrono::steady_clock::now();
  const auto serial = moe::forward_serial(config, model, tokens, trace_pointer);
  const auto serial_end = std::chrono::steady_clock::now();
  const auto reference = moe::forward_naive_reference(config, model, tokens, trace_pointer);

  double squared_error = 0.0;
  double squared_reference = 0.0;
  float max_abs_error = 0.0F;
  for (std::size_t index = 0; index < serial.output.size(); ++index) {
    const double difference = static_cast<double>(serial.output[index]) - reference.output[index];
    squared_error += difference * difference;
    squared_reference += static_cast<double>(reference.output[index]) * reference.output[index];
    max_abs_error = std::max(max_abs_error, static_cast<float>(std::abs(difference)));
  }

  std::uint64_t differing_expert_ids = 0;
  float max_gate_difference = 0.0F;
  for (std::size_t token = 0; token < serial.routes.size(); ++token) {
    for (std::size_t slot = 0; slot < 2; ++slot) {
      differing_expert_ids += serial.routes[token].expert_ids[slot] !=
                               reference.routes[token].expert_ids[slot];
      max_gate_difference = std::max(
          max_gate_difference,
          std::abs(serial.routes[token].gate_weights[slot] - reference.routes[token].gate_weights[slot]));
    }
  }

  const auto histogram = moe::routing_histogram(serial.routes, config.experts);
  const auto elapsed = std::chrono::duration<double>(serial_end - start).count();
  std::cout << "serial forward\n"
            << "  backend=cpu, tokens=" << config.tokens << ", d=" << config.dimension
            << ", h=" << config.hidden_dimension << ", experts=" << config.experts << '\n'
            << "  forward_seconds=" << elapsed << '\n'
            << "  max_abs_error_vs_naive=" << max_abs_error << '\n'
            << "  relative_l2_error_vs_naive="
            << std::sqrt(squared_error / std::max(squared_reference, std::numeric_limits<double>::min()))
            << '\n'
            << "  differing_route_expert_ids=" << differing_expert_ids << '\n'
            << "  max_route_gate_difference=" << max_gate_difference << '\n'
            << "  expert_histogram=";
  for (std::size_t expert = 0; expert < histogram.size(); ++expert) {
    std::cout << (expert == 0 ? "" : ",") << histogram[expert];
  }
  std::cout << '\n';

  const bool routes_match = differing_expert_ids == 0;
  const bool outputs_finite = finite_output(serial.output) && finite_output(reference.output);
  const bool within_tolerance =
      max_abs_error <= 1.0e-4F &&
      std::sqrt(squared_error / std::max(squared_reference, std::numeric_limits<double>::min())) <= 1.0e-5;
  if (routes_match && outputs_finite && within_tolerance) {
    std::cout << "RESULT: PASS\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "RESULT: FAIL\n";
  return EXIT_FAILURE;
}
