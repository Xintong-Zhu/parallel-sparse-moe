#include "moe/protocol.hpp"

#include <limits>

namespace moe {
namespace {

constexpr std::uint64_t kExpertShift = 32;
constexpr std::uint64_t kSlotShift = 48;
constexpr std::uint64_t kExpertMask = 0xffffULL;
constexpr std::uint64_t kSlotMask = 0xffULL;

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

}  // namespace

TokenRange contiguous_token_range(std::uint64_t global_tokens, int rank, int world_size) {
  if (rank < 0 || world_size <= 0 || rank >= world_size) {
    return {};
  }
  const auto world = static_cast<std::uint64_t>(world_size);
  const auto rank_u64 = static_cast<std::uint64_t>(rank);
  const auto base = global_tokens / world;
  const auto remainder = global_tokens % world;
  const auto count = base + (rank_u64 < remainder ? 1 : 0);
  const auto begin = rank_u64 * base + (rank_u64 < remainder ? rank_u64 : remainder);
  return {begin, count};
}

int round_robin_expert_owner(std::uint32_t expert_id, int world_size) {
  if (world_size <= 0) {
    return -1;
  }
  return static_cast<int>(expert_id % static_cast<std::uint32_t>(world_size));
}

WireAssignmentHeader pack_header(const AssignmentHeader& header) {
  WireAssignmentHeader wire;
  wire.word[0] = header.global_token_id;
  wire.word[1] = static_cast<std::uint64_t>(header.source_rank) |
                 (static_cast<std::uint64_t>(header.expert_id) << kExpertShift) |
                 (static_cast<std::uint64_t>(header.slot_id) << kSlotShift);
  return wire;
}

AssignmentHeader unpack_header(const WireAssignmentHeader& wire) {
  AssignmentHeader header;
  header.global_token_id = wire.word[0];
  header.source_rank = static_cast<std::uint32_t>(wire.word[1]);
  header.expert_id = static_cast<std::uint32_t>((wire.word[1] >> kExpertShift) & kExpertMask);
  header.slot_id = static_cast<std::uint8_t>((wire.word[1] >> kSlotShift) & kSlotMask);
  return header;
}

bool valid_assignment_header(const AssignmentHeader& header, const MoEConfig& config,
                             int world_size, std::string* error) {
  if (world_size <= 0) {
    set_error(error, "world size must be positive");
    return false;
  }
  if (header.global_token_id >= config.tokens) {
    set_error(error, "assignment token ID is outside the global token range");
    return false;
  }
  if (header.source_rank >= static_cast<std::uint32_t>(world_size)) {
    set_error(error, "assignment source rank is outside the MPI world");
    return false;
  }
  if (header.expert_id >= config.experts) {
    set_error(error, "assignment expert ID is outside the configured expert range");
    return false;
  }
  if (header.slot_id >= config.top_k) {
    set_error(error, "assignment slot ID is outside the configured top-k range");
    return false;
  }
  return true;
}

bool valid_assignment_buffer(const AssignmentBuffer& buffer, const MoEConfig& config,
                             int world_size, std::string* error) {
  const auto dimension = static_cast<std::size_t>(config.dimension);
  if (buffer.headers.size() > std::numeric_limits<std::size_t>::max() / dimension ||
      buffer.token_values.size() != buffer.headers.size() * dimension) {
    set_error(error, "assignment buffer does not contain exactly d floats per header");
    return false;
  }
  for (const auto& wire : buffer.headers) {
    if (!valid_assignment_header(unpack_header(wire), config, world_size, error)) {
      return false;
    }
  }
  return true;
}

}  // namespace moe
