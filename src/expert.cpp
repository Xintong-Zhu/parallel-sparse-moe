#include "moe/expert.hpp"

#include <algorithm>
#include <stdexcept>

namespace moe {
namespace {

void verify_shape(const ExpertWeights& weights, const MoEConfig& config) {
  const auto w1_size = static_cast<std::size_t>(config.dimension) * config.hidden_dimension;
  const auto w2_size = static_cast<std::size_t>(config.hidden_dimension) * config.dimension;
  if (weights.w1.size() != w1_size || weights.w2.size() != w2_size) {
    throw std::invalid_argument("Expert-weight shape does not match configuration");
  }
}

}  // namespace

void expert_forward_cpu(const ExpertWeights& weights, const MoEConfig& config,
                        const float* token, float* output,
                        std::vector<float>& hidden_scratch) {
  verify_shape(weights, config);
  hidden_scratch.assign(config.hidden_dimension, 0.0F);
  for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
    const float token_value = token[feature];
    const float* row = weights.w1.data() + static_cast<std::size_t>(feature) * config.hidden_dimension;
    for (std::uint32_t hidden = 0; hidden < config.hidden_dimension; ++hidden) {
      hidden_scratch[hidden] += token_value * row[hidden];
    }
  }
  for (auto& value : hidden_scratch) {
    value = std::max(value, 0.0F);
  }
  std::fill(output, output + config.dimension, 0.0F);
  for (std::uint32_t hidden = 0; hidden < config.hidden_dimension; ++hidden) {
    const float hidden_value = hidden_scratch[hidden];
    const float* row = weights.w2.data() + static_cast<std::size_t>(hidden) * config.dimension;
    for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
      output[feature] += hidden_value * row[feature];
    }
  }
}

void expert_forward_cpu_batch(const ExpertWeights& weights, const MoEConfig& config,
                              const float* tokens, std::size_t count, float* outputs,
                              std::vector<float>& hidden_scratch) {
  verify_shape(weights, config);
  if (count == 0) return;
  const std::size_t d = config.dimension;
  const std::size_t h = config.hidden_dimension;
  hidden_scratch.assign(count * h, 0.0F);
  std::fill(outputs, outputs + count * d, 0.0F);
  for (std::size_t feature = 0; feature < d; ++feature) {
    const float* row = weights.w1.data() + feature * h;
    for (std::size_t token = 0; token < count; ++token) {
      const float value = tokens[token * d + feature];
      float* hidden_row = hidden_scratch.data() + token * h;
      for (std::size_t hidden = 0; hidden < h; ++hidden) {
        hidden_row[hidden] += value * row[hidden];
      }
    }
  }
  for (auto& value : hidden_scratch) value = std::max(value, 0.0F);
  for (std::size_t hidden = 0; hidden < h; ++hidden) {
    const float* row = weights.w2.data() + hidden * d;
    for (std::size_t token = 0; token < count; ++token) {
      const float value = hidden_scratch[token * h + hidden];
      float* output_row = outputs + token * d;
      for (std::size_t feature = 0; feature < d; ++feature) {
        output_row[feature] += value * row[feature];
      }
    }
  }
}

void expert_forward_reference(const ExpertWeights& weights, const MoEConfig& config,
                              const float* token, float* output,
                              std::vector<double>& hidden_scratch) {
  verify_shape(weights, config);
  hidden_scratch.assign(config.hidden_dimension, 0.0);
  for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
    const double token_value = token[feature];
    const float* row = weights.w1.data() + static_cast<std::size_t>(feature) * config.hidden_dimension;
    for (std::uint32_t hidden = 0; hidden < config.hidden_dimension; ++hidden) {
      hidden_scratch[hidden] += token_value * static_cast<double>(row[hidden]);
    }
  }
  for (auto& value : hidden_scratch) {
    value = std::max(value, 0.0);
  }
  for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
    double value = 0.0;
    for (std::uint32_t hidden = 0; hidden < config.hidden_dimension; ++hidden) {
      value += hidden_scratch[hidden] *
               static_cast<double>(weights.w2[static_cast<std::size_t>(hidden) *
                                              config.dimension + feature]);
    }
    output[feature] = static_cast<float>(value);
  }
}

}  // namespace moe
