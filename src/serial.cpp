#include "moe/serial.hpp"

#include "moe/expert.hpp"
#include "moe/router.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace moe {
namespace {

ForwardResult forward_impl(const MoEConfig& config, const Model& model,
                           const std::vector<float>& tokens,
                           const std::vector<RoutingDecision>* trace_routes,
                           bool reference) {
  const auto expected_token_values =
      static_cast<std::size_t>(config.tokens) * config.dimension;
  if (tokens.size() != expected_token_values) {
    throw std::invalid_argument("serial forward requires all configured Tokens");
  }
  if ((config.router_mode == RouterMode::kTrace) != (trace_routes != nullptr)) {
    throw std::invalid_argument("trace routes must be present exactly in trace mode");
  }
  if (trace_routes != nullptr && trace_routes->size() != config.tokens) {
    throw std::invalid_argument("trace route count does not match configured Tokens");
  }

  ForwardResult result;
  result.output.resize(expected_token_values);
  result.routes.resize(config.tokens);
  std::vector<float> first_output(config.dimension);
  std::vector<float> second_output(config.dimension);
  std::vector<double> double_hidden;

  for (std::uint64_t token_id = 0; token_id < config.tokens; ++token_id) {
    const auto token_offset = static_cast<std::size_t>(token_id) * config.dimension;
    const float* token = tokens.data() + token_offset;
    const RoutingDecision route =
        trace_routes != nullptr
            ? (*trace_routes)[static_cast<std::size_t>(token_id)]
            : (reference ? route_token_reference(model, config, token)
                         : route_token(model, config, token));
    if (route.expert_ids[0] == route.expert_ids[1] ||
        route.expert_ids[0] >= config.experts || route.expert_ids[1] >= config.experts) {
      throw std::invalid_argument("Token route does not contain two distinct valid Experts");
    }
    result.routes[static_cast<std::size_t>(token_id)] = route;
    if (!reference) continue;
    expert_forward_reference(model.experts[route.expert_ids[0]], config, token,
                             first_output.data(), double_hidden);
    expert_forward_reference(model.experts[route.expert_ids[1]], config, token,
                             second_output.data(), double_hidden);
    float* combined = result.output.data() + token_offset;
    for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
      combined[feature] = route.gate_weights[0] * first_output[feature] +
                          route.gate_weights[1] * second_output[feature];
    }
  }
  if (reference) return result;

  constexpr std::size_t micro_batch = 8;
  const std::size_t d = config.dimension;
  std::vector<std::vector<std::size_t>> by_expert(config.experts);
  for (std::size_t token = 0; token < result.routes.size(); ++token) {
    for (std::size_t slot = 0; slot < 2; ++slot) {
      by_expert[result.routes[token].expert_ids[slot]].push_back(token * 2 + slot);
    }
  }
  std::vector<float> slots(expected_token_values * 2);
  std::vector<float> batch_tokens(micro_batch * d);
  std::vector<float> batch_outputs(micro_batch * d);
  std::vector<float> hidden_scratch;
  for (std::size_t expert = 0; expert < by_expert.size(); ++expert) {
    const auto& assignments = by_expert[expert];
    for (std::size_t begin = 0; begin < assignments.size(); begin += micro_batch) {
      const std::size_t count = std::min(micro_batch, assignments.size() - begin);
      for (std::size_t item = 0; item < count; ++item) {
        const std::size_t token = assignments[begin + item] / 2;
        std::copy_n(tokens.data() + token * d, d, batch_tokens.data() + item * d);
      }
      expert_forward_cpu_batch(model.experts[expert], config, batch_tokens.data(),
                               count, batch_outputs.data(), hidden_scratch);
      for (std::size_t item = 0; item < count; ++item) {
        std::copy_n(batch_outputs.data() + item * d, d,
                    slots.data() + assignments[begin + item] * d);
      }
    }
  }
  for (std::size_t token = 0; token < result.routes.size(); ++token) {
    const auto& route = result.routes[token];
    for (std::size_t feature = 0; feature < d; ++feature) {
      result.output[token * d + feature] =
          route.gate_weights[0] * slots[(token * 2) * d + feature] +
          route.gate_weights[1] * slots[(token * 2 + 1) * d + feature];
    }
  }
  return result;
}

}  // namespace

ForwardResult forward_serial(const MoEConfig& config, const Model& model,
                             const std::vector<float>& tokens,
                             const std::vector<RoutingDecision>* trace_routes) {
  return forward_impl(config, model, tokens, trace_routes, false);
}

ForwardResult forward_naive_reference(
    const MoEConfig& config, const Model& model, const std::vector<float>& tokens,
    const std::vector<RoutingDecision>* trace_routes) {
  return forward_impl(config, model, tokens, trace_routes, true);
}

}  // namespace moe
