#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/serial.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  const auto parsed = moe::parse_command_line(std::vector<std::string>(argv + 1, argv + argc));
  if (!parsed.ok || parsed.help_requested ||
      parsed.config.expert_backend != moe::ExpertBackend::kCpu) {
    std::cerr << "Use valid CPU serial options.\n";
    return 2;
  }
  const auto& config = parsed.config;
  const auto model = moe::generate_model(config);
  const auto tokens = moe::generate_tokens(config, 0, config.tokens);
  const auto trace = config.router_mode == moe::RouterMode::kTrace
                         ? moe::generate_controlled_trace(config)
                         : std::vector<moe::RoutingDecision>{};
  const auto* routes = config.router_mode == moe::RouterMode::kTrace ? &trace : nullptr;
  std::cout << "serial_config,tokens=" << config.tokens << ",experts=" << config.experts
            << ",dimension=" << config.dimension << ",hidden=" << config.hidden_dimension
            << ",router_mode=" << moe::to_string(config.router_mode)
            << ",skew=" << moe::to_string(config.skew)
            << ",model_seed=" << config.model_seed << ",input_seed=" << config.input_seed
            << ",routing_seed=" << config.routing_seed << '\n';
  constexpr int warmups = 3, repeats = 5;
  std::vector<float> expected;
  for (int i = 0; i < warmups + repeats; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = moe::forward_serial(config, model, tokens, routes);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (expected.empty()) expected = result.output;
    bool valid = result.output.size() == expected.size() &&
                 std::memcmp(result.output.data(), expected.data(),
                             expected.size() * sizeof(float)) == 0;
    for (float value : result.output) valid &= std::isfinite(value);
    if (!valid) {
      std::cerr << "serial repeat correctness: FAIL\n";
      return 1;
    }
    if (i >= warmups) {
      std::cout << std::setprecision(9) << "serial_benchmark,trial=" << (i - warmups)
                << ",tokens=" << config.tokens << ",experts=" << config.experts
                << ",end_to_end_seconds=" << seconds << ",bitwise_ok=1\n";
    }
  }
  return 0;
}
