#include "moe/config.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace moe {
namespace {

template <typename Enum>
bool parse_enum(const std::string& value, const std::vector<std::pair<std::string, Enum>>& map,
                Enum& result) {
  for (const auto& item : map) {
    if (item.first == value) {
      result = item.second;
      return true;
    }
  }
  return false;
}

bool parse_u64(const std::string& text, std::uint64_t& value) {
  try {
    std::size_t parsed = 0;
    const auto result = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
      return false;
    }
    value = result;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool parse_u32(const std::string& text, std::uint32_t& value) {
  std::uint64_t parsed = 0;
  if (!parse_u64(text, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  value = static_cast<std::uint32_t>(parsed);
  return true;
}

bool require_number(const std::vector<std::string>& arguments, std::size_t& index,
                    std::uint64_t& destination, std::string& error) {
  if (index + 1 >= arguments.size()) {
    error = arguments[index] + " requires a value";
    return false;
  }
  const std::string& text = arguments[++index];
  const bool parsed = parse_u64(text, destination);
  if (!parsed) {
    error = "invalid unsigned integer for " + arguments[index - 1] + ": " + text;
  }
  return parsed;
}

bool require_number(const std::vector<std::string>& arguments, std::size_t& index,
                    std::uint32_t& destination, std::string& error) {
  if (index + 1 >= arguments.size()) {
    error = arguments[index] + " requires a value";
    return false;
  }
  const std::string& text = arguments[++index];
  const bool parsed = parse_u32(text, destination);
  if (!parsed) {
    error = "invalid unsigned integer for " + arguments[index - 1] + ": " + text;
  }
  return parsed;
}

}  // namespace

const char* to_string(Parallelism value) {
  return value == Parallelism::kExpertParallel ? "ep" : "dp";
}

const char* to_string(ExpertBackend value) {
  return value == ExpertBackend::kCpu ? "cpu" : "cuda";
}

const char* to_string(RouterMode value) {
  return value == RouterMode::kRouter ? "router" : "trace";
}

const char* to_string(Skew value) {
  switch (value) {
    case Skew::kUniform:
      return "uniform";
    case Skew::kModerate:
      return "moderate";
    case Skew::kExtreme:
      return "extreme";
  }
  return "unknown";
}

const char* to_string(HotLayout value) {
  switch (value) {
    case HotLayout::kA:
      return "A";
    case HotLayout::kB:
      return "B";
    case HotLayout::kR1:
      return "R1";
    case HotLayout::kR2:
      return "R2";
    case HotLayout::kR3:
      return "R3";
  }
  return "unknown";
}

const char* to_string(Placement value) {
  return value == Placement::kRoundRobin ? "rr" : "la";
}

const char* to_string(Communication value) {
  return value == Communication::kAlltoallv ? "alltoallv" : "p2p";
}

const char* to_string(OpenMPSchedule value) {
  return value == OpenMPSchedule::kStatic ? "static" : "dynamic";
}

std::string validate_config(const MoEConfig& config) {
  if (config.tokens == 0) {
    return "--tokens must be positive";
  }
  if (config.dimension == 0 || config.hidden_dimension == 0 || config.experts == 0) {
    return "--dim, --hidden, and --experts must be positive";
  }
  if (config.top_k != 2) {
    return "this frozen protocol requires --topk 2";
  }
  if (config.experts < config.top_k) {
    return "--experts must be at least --topk";
  }
  if (config.experts > std::numeric_limits<std::uint16_t>::max()) {
    return "--experts must fit in the 16-bit assignment wire field";
  }
  if (config.router_mode == RouterMode::kRouter && config.skew != Skew::kUniform) {
    return "controlled skew requires --router-mode trace";
  }
  if (config.router_mode == RouterMode::kTrace && config.skew != Skew::kUniform &&
      config.experts != 16) {
    return "the frozen moderate and extreme trace layouts require --experts 16";
  }
  return {};
}

ParseResult parse_command_line(const std::vector<std::string>& arguments) {
  ParseResult result;
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    const std::string& argument = arguments[index];
    if (argument == "--help" || argument == "-h") {
      result.ok = true;
      result.help_requested = true;
      return result;
    }
    if (argument == "--tokens") {
      if (!require_number(arguments, index, result.config.tokens, result.error)) return result;
    } else if (argument == "--dim") {
      if (!require_number(arguments, index, result.config.dimension, result.error)) return result;
    } else if (argument == "--hidden") {
      if (!require_number(arguments, index, result.config.hidden_dimension, result.error)) return result;
    } else if (argument == "--experts") {
      if (!require_number(arguments, index, result.config.experts, result.error)) return result;
    } else if (argument == "--topk") {
      if (!require_number(arguments, index, result.config.top_k, result.error)) return result;
    } else if (argument == "--model-seed") {
      if (!require_number(arguments, index, result.config.model_seed, result.error)) return result;
    } else if (argument == "--input-seed") {
      if (!require_number(arguments, index, result.config.input_seed, result.error)) return result;
    } else if (argument == "--routing-seed") {
      if (!require_number(arguments, index, result.config.routing_seed, result.error)) return result;
    } else if (argument == "--calibration-seed") {
      if (!require_number(arguments, index, result.config.calibration_seed, result.error)) return result;
    } else {
      if (index + 1 >= arguments.size()) {
        result.error = argument + " requires a value";
        return result;
      }
      const std::string& value = arguments[++index];
      if (argument == "--parallelism") {
        if (!parse_enum(value, {{"ep", Parallelism::kExpertParallel}, {"dp", Parallelism::kDataParallel}},
                        result.config.parallelism)) {
          result.error = "--parallelism must be ep or dp";
          return result;
        }
      } else if (argument == "--backend") {
        if (!parse_enum(value, {{"cpu", ExpertBackend::kCpu}, {"cuda", ExpertBackend::kCuda}},
                        result.config.expert_backend)) {
          result.error = "--backend must be cpu or cuda";
          return result;
        }
      } else if (argument == "--router-mode") {
        if (!parse_enum(value, {{"router", RouterMode::kRouter}, {"trace", RouterMode::kTrace}},
                        result.config.router_mode)) {
          result.error = "--router-mode must be router or trace";
          return result;
        }
      } else if (argument == "--skew") {
        if (!parse_enum(value, {{"uniform", Skew::kUniform}, {"moderate", Skew::kModerate},
                                {"extreme", Skew::kExtreme}}, result.config.skew)) {
          result.error = "--skew must be uniform, moderate, or extreme";
          return result;
        }
      } else if (argument == "--hot-layout") {
        if (!parse_enum(value, {{"A", HotLayout::kA}, {"B", HotLayout::kB}, {"R1", HotLayout::kR1},
                                {"R2", HotLayout::kR2}, {"R3", HotLayout::kR3}},
                        result.config.hot_layout)) {
          result.error = "--hot-layout must be A, B, R1, R2, or R3";
          return result;
        }
      } else if (argument == "--placement") {
        if (!parse_enum(value, {{"rr", Placement::kRoundRobin}, {"la", Placement::kLoadAware}},
                        result.config.placement)) {
          result.error = "--placement must be rr or la";
          return result;
        }
      } else if (argument == "--communication") {
        if (!parse_enum(value, {{"alltoallv", Communication::kAlltoallv},
                                {"p2p", Communication::kPointToPoint}},
                        result.config.communication)) {
          result.error = "--communication must be alltoallv or p2p";
          return result;
        }
      } else if (argument == "--omp-schedule") {
        if (!parse_enum(value, {{"static", OpenMPSchedule::kStatic},
                                {"dynamic", OpenMPSchedule::kDynamic}},
                        result.config.omp_schedule)) {
          result.error = "--omp-schedule must be static or dynamic";
          return result;
        }
      } else {
        result.error = "unknown option: " + argument;
        return result;
      }
    }
  }
  result.error = validate_config(result.config);
  result.ok = result.error.empty();
  return result;
}

std::string command_line_help() {
  return R"(Usage: moe_contract_check [options]

Frozen model options:
  --tokens N --dim D --hidden H --experts E --topk 2
  --model-seed S --input-seed S --routing-seed S --calibration-seed S

Parallel protocol options:
  --parallelism ep|dp --backend cpu|cuda --router-mode router|trace
  --skew uniform|moderate|extreme --hot-layout A|B|R1|R2|R3
  --placement rr|la --communication alltoallv|p2p
  --omp-schedule static|dynamic
)";
}

}  // namespace moe
