#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moe {

enum class Parallelism { kExpertParallel, kDataParallel };
enum class RouterMode { kRouter, kTrace };
enum class Skew { kUniform, kModerate, kExtreme };
enum class HotLayout { kA, kB, kR1, kR2, kR3 };
enum class Placement { kRoundRobin, kLoadAware };
enum class Communication { kAlltoallv, kPointToPoint };
enum class OpenMPSchedule { kStatic, kDynamic };

struct MoEConfig {
  std::uint64_t tokens = 2048;
  std::uint32_t dimension = 128;
  std::uint32_t hidden_dimension = 256;
  std::uint32_t experts = 16;
  std::uint32_t top_k = 2;

  std::uint64_t model_seed = 42;
  std::uint64_t input_seed = 314159;
  std::uint64_t routing_seed = 271828;
  std::uint64_t calibration_seed = 161803;

  Parallelism parallelism = Parallelism::kExpertParallel;
  RouterMode router_mode = RouterMode::kRouter;
  Skew skew = Skew::kUniform;
  HotLayout hot_layout = HotLayout::kA;
  Placement placement = Placement::kRoundRobin;
  Communication communication = Communication::kAlltoallv;
  OpenMPSchedule omp_schedule = OpenMPSchedule::kStatic;
};

struct ParseResult {
  bool ok = false;
  bool help_requested = false;
  MoEConfig config;
  std::string error;
};

const char* to_string(Parallelism value);
const char* to_string(RouterMode value);
const char* to_string(Skew value);
const char* to_string(HotLayout value);
const char* to_string(Placement value);
const char* to_string(Communication value);
const char* to_string(OpenMPSchedule value);

ParseResult parse_command_line(const std::vector<std::string>& arguments);
std::string validate_config(const MoEConfig& config);
std::string command_line_help();

}  // namespace moe
