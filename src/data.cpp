#include "moe/data.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace moe {
namespace {

constexpr std::uint64_t kTokenDomain = 0x91f2b3c4d5e60718ULL;
constexpr std::uint64_t kRouterDomain = 0x2a4c6e8f10293847ULL;
constexpr std::uint64_t kW1Domain = 0x7b9d1f3e5a6c8d0fULL;
constexpr std::uint64_t kW2Domain = 0x13579bdf2468ace0ULL;
constexpr std::uint64_t kTraceTieDomain = 0x8c6a4e2d0f1b3957ULL;
constexpr std::uint64_t kTraceGateDomain = 0x4f3d2b1a09e8d7c6ULL;
constexpr double kTwoPi = 6.283185307179586476925286766559;

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t keyed_u64(std::uint64_t seed, std::uint64_t domain,
                        std::uint64_t index) {
  return splitmix64(seed ^ splitmix64(domain) ^ splitmix64(index));
}

double uniform_open01(std::uint64_t seed, std::uint64_t domain,
                      std::uint64_t index) {
  // Use the high 53 bits so conversion is stable and excludes zero and one.
  constexpr double kScale = 1.0 / 9007199254740992.0;  // 2^53
  const auto bits = keyed_u64(seed, domain, index) >> 11U;
  return (static_cast<double>(bits) + 0.5) * kScale;
}

float normal_value(std::uint64_t seed, std::uint64_t domain, std::uint64_t index,
                   double standard_deviation) {
  const double u1 = uniform_open01(seed, domain, 2U * index);
  const double u2 = uniform_open01(seed, domain, 2U * index + 1U);
  const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
  return static_cast<float>(z * standard_deviation);
}

std::vector<std::uint64_t> allocate_counts(std::uint64_t total,
                                           const std::vector<double>& shares) {
  std::vector<std::uint64_t> counts(shares.size(), 0);
  std::vector<std::pair<double, std::size_t>> fractional;
  fractional.reserve(shares.size());
  std::uint64_t assigned = 0;
  for (std::size_t expert = 0; expert < shares.size(); ++expert) {
    const double exact = static_cast<double>(total) * shares[expert];
    counts[expert] = static_cast<std::uint64_t>(std::floor(exact));
    assigned += counts[expert];
    fractional.emplace_back(exact - static_cast<double>(counts[expert]), expert);
  }
  std::sort(fractional.begin(), fractional.end(),
            [](const auto& left, const auto& right) {
              return left.first == right.first ? left.second < right.second
                                               : left.first > right.first;
            });
  for (std::uint64_t i = 0; i < total - assigned; ++i) {
    ++counts[fractional[static_cast<std::size_t>(i % fractional.size())].second];
  }
  return counts;
}

std::vector<std::uint32_t> hot_experts(const MoEConfig& config) {
  if (config.experts != 16U) {
    throw std::invalid_argument(
        "moderate and extreme trace workloads require the frozen E=16 configuration");
  }
  switch (config.hot_layout) {
    case HotLayout::kA:
      return {0U, 4U, 8U, 12U};
    case HotLayout::kB:
      return {0U, 1U, 2U, 3U};
    case HotLayout::kR1:
      return {1U, 6U, 9U, 14U};
    case HotLayout::kR2:
      return {2U, 5U, 11U, 15U};
    case HotLayout::kR3:
      return {3U, 7U, 10U, 13U};
  }
  throw std::logic_error("unknown hot Expert layout");
}

std::vector<double> trace_shares(const MoEConfig& config) {
  std::vector<double> shares(config.experts, 1.0 / static_cast<double>(config.experts));
  if (config.skew == Skew::kUniform) {
    return shares;
  }

  if (config.skew == Skew::kModerate) {
    std::fill(shares.begin(), shares.end(), 1.0 / 30.0);
    for (const auto expert : hot_experts(config)) {
      shares[expert] = 0.15;
    }
    return shares;
  }

  if (config.experts != 16U) {
    throw std::invalid_argument("extreme trace workload requires the frozen E=16 configuration");
  }
  std::fill(shares.begin(), shares.end(), 0.04);
  shares[0] = 0.40;
  return shares;
}

}  // namespace

static Model generate_model_impl(const MoEConfig& config, int rank, int world_size,
                                 bool partitioned, const std::vector<int>* owners = nullptr) {
  const std::size_t d = config.dimension;
  const std::size_t h = config.hidden_dimension;
  const std::size_t e = config.experts;
  const double router_stddev = std::sqrt(1.0 / static_cast<double>(d));
  const double w1_stddev = std::sqrt(1.0 / static_cast<double>(d));
  const double w2_stddev = std::sqrt(1.0 / static_cast<double>(h));

  Model model;
  model.router_weights.resize(d * e);
  for (std::size_t index = 0; index < model.router_weights.size(); ++index) {
    model.router_weights[index] = normal_value(config.model_seed, kRouterDomain, index,
                                                router_stddev);
  }

  model.experts.resize(e);
  for (std::size_t expert = 0; expert < e; ++expert) {
    const int owner = owners == nullptr
                          ? static_cast<int>(expert % static_cast<std::size_t>(world_size))
                          : (*owners)[expert];
    if (partitioned && owner != rank) {
      continue;
    }
    auto& weights = model.experts[expert];
    weights.w1.resize(d * h);
    weights.w2.resize(h * d);
    const auto w1_base = static_cast<std::uint64_t>(expert) * weights.w1.size();
    const auto w2_base = static_cast<std::uint64_t>(expert) * weights.w2.size();
    for (std::size_t index = 0; index < weights.w1.size(); ++index) {
      weights.w1[index] = normal_value(config.model_seed, kW1Domain,
                                       w1_base + index, w1_stddev);
    }
    for (std::size_t index = 0; index < weights.w2.size(); ++index) {
      weights.w2[index] = normal_value(config.model_seed, kW2Domain,
                                       w2_base + index, w2_stddev);
    }
  }
  return model;
}

Model generate_model(const MoEConfig& config) {
  return generate_model_impl(config, 0, 1, false);
}

Model generate_round_robin_partitioned_model(const MoEConfig& config, int rank,
                                              int world_size) {
  if (world_size <= 0 || rank < 0 || rank >= world_size) {
    throw std::invalid_argument("invalid rank or world size for partitioned model");
  }
  return generate_model_impl(config, rank, world_size, true);
}

Model generate_partitioned_model(const MoEConfig& config, int rank,
                                 int world_size, const std::vector<int>& owners) {
  if (world_size <= 0 || rank < 0 || rank >= world_size ||
      owners.size() != config.experts ||
      std::any_of(owners.begin(), owners.end(), [world_size](int owner) {
        return owner < 0 || owner >= world_size;
      })) {
    throw std::invalid_argument("invalid owner map for partitioned model");
  }
  return generate_model_impl(config, rank, world_size, true, &owners);
}

std::vector<float> generate_tokens(const MoEConfig& config, std::uint64_t first_token,
                                   std::uint64_t token_count) {
  if (first_token > config.tokens || token_count > config.tokens - first_token) {
    throw std::out_of_range("requested Token range is outside the configured workload");
  }
  const std::size_t d = config.dimension;
  if (token_count > std::numeric_limits<std::size_t>::max() / d) {
    throw std::overflow_error("Token buffer is too large for this process");
  }
  std::vector<float> tokens(static_cast<std::size_t>(token_count) * d);
  for (std::uint64_t local_token = 0; local_token < token_count; ++local_token) {
    const auto global_token = first_token + local_token;
    for (std::size_t feature = 0; feature < d; ++feature) {
      const auto index = global_token * d + feature;
      tokens[static_cast<std::size_t>(local_token) * d + feature] =
          normal_value(config.input_seed, kTokenDomain, index, 1.0);
    }
  }
  return tokens;
}

std::vector<RoutingDecision> generate_controlled_trace(const MoEConfig& config) {
  if (config.router_mode != RouterMode::kTrace) {
    throw std::invalid_argument("controlled traces require --router-mode trace");
  }
  if (config.tokens > std::numeric_limits<std::uint64_t>::max() / 2U) {
    throw std::overflow_error("2N assignments overflow uint64_t");
  }

  // Large uniform Expert sets need O(N) construction for scaling experiments.
  // Each complete cycle assigns every Expert exactly once; a partial cycle
  // adds at most one assignment to any Expert. The seeded permutation and
  // per-cycle rotation vary pairings without changing these exact counts.
  if (config.skew == Skew::kUniform && config.experts >= 64U &&
      config.experts % 2U == 0U) {
    std::vector<std::uint32_t> permutation(config.experts);
    std::iota(permutation.begin(), permutation.end(), 0U);
    std::sort(permutation.begin(), permutation.end(), [&](std::uint32_t left,
                                                          std::uint32_t right) {
      const auto left_key = keyed_u64(config.routing_seed, kTraceTieDomain, left);
      const auto right_key = keyed_u64(config.routing_seed, kTraceTieDomain, right);
      return left_key == right_key ? left < right : left_key < right_key;
    });
    const std::uint64_t cycle_tokens = config.experts / 2U;
    std::vector<RoutingDecision> routes(config.tokens);
    for (std::uint64_t token = 0; token < config.tokens; ++token) {
      const auto cycle = token / cycle_tokens;
      const auto position = token % cycle_tokens;
      const auto rotation = keyed_u64(config.routing_seed, kTraceTieDomain,
                                      cycle + config.experts) % config.experts;
      const auto first_index = (rotation + 2U * position) % config.experts;
      const auto second_index = (first_index + 1U) % config.experts;
      const float first_gate = static_cast<float>(
          0.5 + 0.5 * uniform_open01(config.routing_seed, kTraceGateDomain, token));
      routes[static_cast<std::size_t>(token)] =
          {{permutation[static_cast<std::size_t>(first_index)],
            permutation[static_cast<std::size_t>(second_index)]},
           {first_gate, 1.0F - first_gate}};
    }
    return routes;
  }

  auto remaining = allocate_counts(2U * config.tokens, trace_shares(config));
  for (const auto count : remaining) {
    if (count > config.tokens) {
      throw std::logic_error("trace Expert assignment count exceeds one assignment per Token");
    }
  }

  std::vector<RoutingDecision> routes(config.tokens);
  for (std::uint64_t token = 0; token < config.tokens; ++token) {
    std::vector<std::uint32_t> ordered(config.experts);
    for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
      ordered[expert] = expert;
    }
    std::sort(ordered.begin(), ordered.end(), [&](std::uint32_t left, std::uint32_t right) {
      if (remaining[left] != remaining[right]) {
        return remaining[left] > remaining[right];
      }
      return keyed_u64(config.routing_seed, kTraceTieDomain,
                       token * config.experts + left) <
             keyed_u64(config.routing_seed, kTraceTieDomain,
                       token * config.experts + right);
    });
    const auto first = ordered[0];
    const auto second = ordered[1];
    if (remaining[first] == 0U || remaining[second] == 0U) {
      throw std::logic_error("trace construction exhausted an Expert before all Tokens");
    }
    --remaining[first];
    --remaining[second];

    const float first_gate = static_cast<float>(
        0.5 + 0.5 * uniform_open01(config.routing_seed, kTraceGateDomain, token));
    routes[static_cast<std::size_t>(token)] = {{first, second}, {first_gate, 1.0F - first_gate}};
  }
  if (std::any_of(remaining.begin(), remaining.end(),
                  [](std::uint64_t count) { return count != 0U; })) {
    throw std::logic_error("trace construction did not consume all Expert assignments");
  }
  return routes;
}

std::vector<std::uint64_t> routing_histogram(const std::vector<RoutingDecision>& routes,
                                              std::uint32_t expert_count) {
  std::vector<std::uint64_t> histogram(expert_count, 0U);
  for (const auto& route : routes) {
    for (const auto expert : route.expert_ids) {
      if (expert >= expert_count) {
        throw std::out_of_range("route contains an Expert outside the configured range");
      }
      ++histogram[expert];
    }
  }
  return histogram;
}

}  // namespace moe
