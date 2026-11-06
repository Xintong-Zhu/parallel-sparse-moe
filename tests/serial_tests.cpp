#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/expert.hpp"
#include "moe/protocol.hpp"
#include "moe/serial.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  for (int rank = 0; rank < 8; ++rank) {
    const auto owned = moe::generate_round_robin_partitioned_model(config, rank, 8);
    ok &= check(owned.router_weights == model_a.router_weights,
                "partitioned model preserves router weights");
    for (std::size_t expert = 0; expert < config.experts; ++expert) {
      const bool local = static_cast<int>(expert % 8) == rank;
      ok &= check(local ? (owned.experts[expert].w1 == model_a.experts[expert].w1 &&
                           owned.experts[expert].w2 == model_a.experts[expert].w2)
                        : (owned.experts[expert].w1.empty() &&
                           owned.experts[expert].w2.empty()),
                  "partitioned model stores only owned Expert weights");
    }
  }

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

  // The batch kernel must preserve the scalar kernel's exact FP32 operation
  // order, including partial batches used by the grouped serial path.
  moe::MoEConfig batch_config = config;
  batch_config.tokens = 9;
  batch_config.dimension = 128;
  batch_config.hidden_dimension = 256;
  const auto batch_model = moe::generate_model(batch_config);
  const auto batch_tokens = moe::generate_tokens(batch_config, 0, batch_config.tokens);
  for (std::size_t count : {std::size_t{1}, std::size_t{4}, std::size_t{7},
                            std::size_t{8}, std::size_t{9}}) {
    std::vector<float> batched(count * batch_config.dimension);
    std::vector<float> scalar(count * batch_config.dimension);
    std::vector<float> batch_scratch, scalar_scratch;
    moe::expert_forward_cpu_batch(batch_model.experts[0], batch_config,
                                  batch_tokens.data(), count, batched.data(), batch_scratch);
    for (std::size_t token = 0; token < count; ++token) {
      moe::expert_forward_cpu(batch_model.experts[0], batch_config,
                              batch_tokens.data() + token * batch_config.dimension,
                              scalar.data() + token * batch_config.dimension, scalar_scratch);
    }
    ok &= check(std::memcmp(batched.data(), scalar.data(),
                            batched.size() * sizeof(float)) == 0,
                "batched and scalar Expert outputs are bitwise identical");
  }

  moe::MoEConfig grouped_config = config;
  grouped_config.tokens = 17;
  const auto grouped_model = moe::generate_model(grouped_config);
  const auto grouped_tokens = moe::generate_tokens(grouped_config, 0, grouped_config.tokens);
  const auto grouped_trace = moe::generate_controlled_trace(grouped_config);
  const auto grouped = moe::forward_serial(grouped_config, grouped_model,
                                            grouped_tokens, &grouped_trace);
  std::vector<float> scalar_output(grouped.output.size());
  std::vector<float> first(grouped_config.dimension), second(grouped_config.dimension), scratch;
  for (std::size_t token = 0; token < grouped_config.tokens; ++token) {
    const auto& route = grouped.routes[token];
    const float* input = grouped_tokens.data() + token * grouped_config.dimension;
    moe::expert_forward_cpu(grouped_model.experts[route.expert_ids[0]], grouped_config,
                            input, first.data(), scratch);
    moe::expert_forward_cpu(grouped_model.experts[route.expert_ids[1]], grouped_config,
                            input, second.data(), scratch);
    for (std::size_t feature = 0; feature < grouped_config.dimension; ++feature) {
      scalar_output[token * grouped_config.dimension + feature] =
          route.gate_weights[0] * first[feature] + route.gate_weights[1] * second[feature];
    }
  }
  ok &= check(std::memcmp(grouped.output.data(), scalar_output.data(),
                          grouped.output.size() * sizeof(float)) == 0,
              "grouped serial output equals per-Token execution bitwise");

  if (!ok) {
    return 1;
  }
  std::cout << "serial tests: PASS\n";
  return 0;
}
