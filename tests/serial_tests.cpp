#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/protocol.hpp"
#include "moe/serial.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

bool vectors_equal(const std::vector<float>& left, const std::vector<float>& right) {
  return left == right;
}

}  // namespace

int main() {
  bool ok = true;

  moe::MoEConfig config;
  config.tokens = 7;
  config.dimension = 4;
  config.hidden_dimension = 8;
  config.experts = 4;
  config.router_mode = moe::RouterMode::kTrace;
  config.skew = moe::Skew::kUniform;

  const auto all_tokens = moe::generate_tokens(config, 0, config.tokens);
  std::vector<float> partitioned;
  const auto range0 = moe::contiguous_token_range(config.tokens, 0, 3);
  const auto range1 = moe::contiguous_token_range(config.tokens, 1, 3);
  const auto range2 = moe::contiguous_token_range(config.tokens, 2, 3);
  for (const auto range : {range0, range1, range2}) {
    const auto part = moe::generate_tokens(config, range.begin, range.count);
    partitioned.insert(partitioned.end(), part.begin(), part.end());
  }
  ok &= check(vectors_equal(all_tokens, partitioned), "token generation is global-index deterministic");

  const auto model_a = moe::generate_model(config);
  const auto model_b = moe::generate_model(config);
  ok &= check(model_a.router_weights == model_b.router_weights,
              "router weights are deterministic");
  ok &= check(model_a.experts[0].w1 == model_b.experts[0].w1 &&
                  model_a.experts[0].w2 == model_b.experts[0].w2,
              "expert weights are deterministic");

  const auto trace = moe::generate_controlled_trace(config);
  const auto histogram = moe::routing_histogram(trace, config.experts);
  std::uint64_t histogram_sum = 0;
  for (std::uint64_t count : histogram) {
    histogram_sum += count;
  }
  ok &= check(histogram_sum == 2 * config.tokens, "trace contains exactly two assignments per token");
  for (const auto& route : trace) {
    ok &= check(route.expert_ids[0] != route.expert_ids[1], "trace experts are distinct");
    ok &= check(std::abs(route.gate_weights[0] + route.gate_weights[1] - 1.0F) < 1.0e-6F,
                "trace gates sum to one");
  }

  const auto serial = moe::forward_serial(config, model_a, all_tokens, &trace);
  const auto reference = moe::forward_naive_reference(config, model_a, all_tokens, &trace);
  float max_abs_error = 0.0F;
  for (std::size_t index = 0; index < serial.output.size(); ++index) {
    ok &= check(std::isfinite(serial.output[index]) && std::isfinite(reference.output[index]),
                "serial and reference outputs are finite");
    max_abs_error = std::max(max_abs_error, std::abs(serial.output[index] - reference.output[index]));
  }
  ok &= check(serial.routes.size() == reference.routes.size(), "route counts agree");
  for (std::size_t token = 0; token < serial.routes.size(); ++token) {
    ok &= check(serial.routes[token].expert_ids == reference.routes[token].expert_ids,
                "trace route IDs agree");
  }
  ok &= check(max_abs_error <= 1.0e-4F, "serial output agrees with FP64 reference");

  if (!ok) {
    return 1;
  }
  std::cout << "serial tests: PASS\n";
  return 0;
}
