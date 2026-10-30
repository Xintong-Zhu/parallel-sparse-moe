#include "moe/router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace moe {
namespace {

template <typename Accumulator>
RoutingDecision route_impl(const Model& model, const MoEConfig& config,
                           const float* token) {
  if (model.router_weights.size() !=
      static_cast<std::size_t>(config.dimension) * config.experts) {
    throw std::invalid_argument("router-weight shape does not match configuration");
  }

  std::uint32_t best = 0;
  std::uint32_t second = 1;
  Accumulator best_score = -std::numeric_limits<Accumulator>::infinity();
  Accumulator second_score = -std::numeric_limits<Accumulator>::infinity();
  for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
    Accumulator score = 0;
    for (std::uint32_t feature = 0; feature < config.dimension; ++feature) {
      score += static_cast<Accumulator>(token[feature]) *
               static_cast<Accumulator>(
                   model.router_weights[static_cast<std::size_t>(feature) * config.experts + expert]);
    }
    if (score > best_score || (score == best_score && expert < best)) {
      second = best;
      second_score = best_score;
      best = expert;
      best_score = score;
    } else if (expert != best &&
               (score > second_score || (score == second_score && expert < second))) {
      second = expert;
      second_score = score;
    }
  }

  const Accumulator max_score = std::max(best_score, second_score);
  const Accumulator first_exp = std::exp(best_score - max_score);
  const Accumulator second_exp = std::exp(second_score - max_score);
  const Accumulator denominator = first_exp + second_exp;
  return {{best, second}, {static_cast<float>(first_exp / denominator),
                           static_cast<float>(second_exp / denominator)}};
}

}  // namespace

RoutingDecision route_token(const Model& model, const MoEConfig& config, const float* token) {
  return route_impl<float>(model, config, token);
}

RoutingDecision route_token_reference(const Model& model, const MoEConfig& config,
                                      const float* token) {
  return route_impl<double>(model, config, token);
}

}  // namespace moe
