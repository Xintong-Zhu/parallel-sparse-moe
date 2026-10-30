#include "moe/serial.hpp"

#include "moe/expert.hpp"
#include "moe/router.hpp"

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
  std::vector<float> float_hidden;
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
    if (reference) {
      expert_forward_reference(model.experts[route.expert_ids[0]], config, token,
                               first_output.data(), double_hidden);
      expert_forward_reference(model.experts[route.expert_ids[1]], config, token,
                               second_output.data(), double_hidden);
    } else {
      expert_forward_cpu(model.experts[route.expert_ids[0]], config, token,
                         first_output.data(), float_hidden);
      expert_forward_cpu(model.experts[route.expert_ids[1]], config, token,
                         second_output.data(), float_hidden);
    }
    float* combined = result.output.data() + token_offset;
    for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
      combined[feature] = route.gate_weights[0] * first_output[feature] +
                          route.gate_weights[1] * second_output[feature];
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
