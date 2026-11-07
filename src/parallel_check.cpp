#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/expert.hpp"
#include "moe/protocol.hpp"
#include "moe/router.hpp"
#include "moe/serial.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Buffers = std::vector<moe::AssignmentBuffer>;
constexpr std::size_t micro_batch = 8;

struct PhaseTiming {
  double dispatch_pack = 0.0;
  double dispatch_comm = 0.0;
  double compute = 0.0;
  double return_pack = 0.0;
  double return_comm = 0.0;
  double validation_unpack = 0.0;
  double combine = 0.0;
};

struct ExpertTask {
  std::uint32_t expert;
  std::size_t begin;
  std::size_t count;
};

std::vector<ExpertTask> make_tasks(
    const std::vector<std::vector<std::size_t>>& by_expert) {
  std::vector<ExpertTask> tasks;
  for (std::size_t expert = 0; expert < by_expert.size(); ++expert) {
    for (std::size_t begin = 0; begin < by_expert[expert].size(); begin += micro_batch) {
      tasks.push_back({static_cast<std::uint32_t>(expert), begin,
                       std::min(micro_batch, by_expert[expert].size() - begin)});
    }
  }
  return tasks;
}

int mpi_count(std::size_t value) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    throw std::overflow_error("MPI message exceeds INT_MAX elements");
  }
  return static_cast<int>(value);
}

// Keep one assignment's header and vector together while grouping by destination.
void append(moe::AssignmentBuffer& destination, const moe::WireAssignmentHeader& header,
            const float* values, std::uint32_t dimension) {
  destination.headers.push_back(header);
  destination.token_values.insert(destination.token_values.end(), values,
                                  values + dimension);
}

moe::AssignmentBuffer exchange(const Buffers& destinations, std::uint32_t dimension,
                               moe::Communication communication, int tag_base,
                               MPI_Comm communicator) {
  int rank = 0, world = 0;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &world);
  if (destinations.size() != static_cast<std::size_t>(world)) {
    throw std::invalid_argument("one send buffer is required per rank");
  }
  std::vector<int> send_counts(world), receive_counts(world);
  std::vector<int> send_offsets(world), receive_offsets(world);
  std::size_t total_send = 0;
  for (int peer = 0; peer < world; ++peer) {
    const auto& part = destinations[peer];
    if (part.token_values.size() != part.headers.size() * dimension) {
      throw std::invalid_argument("malformed outgoing assignment buffer");
    }
    send_counts[peer] = mpi_count(part.headers.size());
    send_offsets[peer] = mpi_count(total_send);
    total_send += part.headers.size();
  }
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT,
               communicator);
  std::size_t total_receive = 0;
  for (int peer = 0; peer < world; ++peer) {
    receive_offsets[peer] = mpi_count(total_receive);
    total_receive += static_cast<std::size_t>(receive_counts[peer]);
  }
  moe::AssignmentBuffer outbound, inbound;
  outbound.headers.reserve(total_send);
  outbound.token_values.reserve(total_send * dimension);
  for (const auto& part : destinations) {
    outbound.headers.insert(outbound.headers.end(), part.headers.begin(),
                            part.headers.end());
    outbound.token_values.insert(outbound.token_values.end(), part.token_values.begin(),
                                 part.token_values.end());
  }
  inbound.headers.resize(total_receive);
  inbound.token_values.resize(total_receive * dimension);
  auto scale = [&](const std::vector<int>& source, std::size_t multiplier) {
    std::vector<int> result(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      result[i] = mpi_count(static_cast<std::size_t>(source[i]) * multiplier);
    }
    return result;
  };
  const auto send_header_counts = scale(send_counts, sizeof(moe::WireAssignmentHeader));
  const auto send_header_offsets = scale(send_offsets, sizeof(moe::WireAssignmentHeader));
  const auto recv_header_counts = scale(receive_counts, sizeof(moe::WireAssignmentHeader));
  const auto recv_header_offsets = scale(receive_offsets, sizeof(moe::WireAssignmentHeader));
  const auto send_value_counts = scale(send_counts, dimension);
  const auto send_value_offsets = scale(send_offsets, dimension);
  const auto recv_value_counts = scale(receive_counts, dimension);
  const auto recv_value_offsets = scale(receive_offsets, dimension);
  if (communication == moe::Communication::kAlltoallv) {
    moe::WireAssignmentHeader empty_header{};
    float empty_value = 0.0F;
    MPI_Alltoallv(outbound.headers.empty() ? &empty_header : outbound.headers.data(),
                  send_header_counts.data(), send_header_offsets.data(), MPI_BYTE,
                  inbound.headers.empty() ? &empty_header : inbound.headers.data(),
                  recv_header_counts.data(), recv_header_offsets.data(), MPI_BYTE,
                  communicator);
    MPI_Alltoallv(outbound.token_values.empty() ? &empty_value : outbound.token_values.data(),
                  send_value_counts.data(), send_value_offsets.data(), MPI_FLOAT,
                  inbound.token_values.empty() ? &empty_value : inbound.token_values.data(),
                  recv_value_counts.data(), recv_value_offsets.data(), MPI_FLOAT,
                  communicator);
  } else {
    if (send_counts[rank] != receive_counts[rank]) {
      throw std::runtime_error("self-send count differs from self-receive count");
    }
    if (send_counts[rank] > 0) {
      std::copy_n(outbound.headers.data() + send_offsets[rank], send_counts[rank],
                  inbound.headers.data() + receive_offsets[rank]);
      std::copy_n(outbound.token_values.data() + send_value_offsets[rank],
                  send_value_counts[rank],
                  inbound.token_values.data() + recv_value_offsets[rank]);
    }
    std::vector<MPI_Request> requests;
    requests.reserve(4U * static_cast<std::size_t>(world - 1));
    // Post every receive before any send; each peer occupies its own slice.
    for (int peer = 0; peer < world; ++peer) {
      if (peer == rank || receive_counts[peer] == 0) continue;
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(inbound.headers.data() + receive_offsets[peer], recv_header_counts[peer],
                MPI_BYTE, peer, tag_base, communicator, &requests.back());
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(inbound.token_values.data() + recv_value_offsets[peer],
                recv_value_counts[peer], MPI_FLOAT, peer, tag_base + 1, communicator,
                &requests.back());
    }
    for (int peer = 0; peer < world; ++peer) {
      if (peer == rank || send_counts[peer] == 0) continue;
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Isend(outbound.headers.data() + send_offsets[peer], send_header_counts[peer],
                MPI_BYTE, peer, tag_base, communicator, &requests.back());
      requests.push_back(MPI_REQUEST_NULL);
      MPI_Isend(outbound.token_values.data() + send_value_offsets[peer],
                send_value_counts[peer], MPI_FLOAT, peer, tag_base + 1, communicator,
                &requests.back());
    }
    if (!requests.empty()) {
      MPI_Waitall(mpi_count(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
  }
  return inbound;
}

std::vector<moe::RoutingDecision> local_routes(
    const moe::MoEConfig& config, const moe::Model& model,
    const std::vector<float>& local_tokens, moe::TokenRange range,
    const std::vector<moe::RoutingDecision>& trace) {
  std::vector<moe::RoutingDecision> routes(range.count);
  for (std::uint64_t local = 0; local < range.count; ++local) {
    routes[local] = config.router_mode == moe::RouterMode::kTrace
                        ? trace[range.begin + local]
                        : moe::route_token(model, config,
                                           local_tokens.data() + local * config.dimension);
  }
  return routes;
}

std::vector<float> forward_dp(const moe::MoEConfig& config, const moe::Model& model,
                              const std::vector<float>& tokens,
                              const std::vector<moe::RoutingDecision>& routes,
                              PhaseTiming* timing = nullptr) {
  const double compute_start = MPI_Wtime();
  std::vector<float> output(tokens.size());
  const std::size_t d = config.dimension;
  std::vector<std::vector<std::size_t>> by_expert(config.experts);
  for (std::size_t token = 0; token < routes.size(); ++token) {
    for (std::size_t slot = 0; slot < 2; ++slot) {
      by_expert[routes[token].expert_ids[slot]].push_back(token * 2 + slot);
    }
  }
  const auto tasks = make_tasks(by_expert);
  std::vector<float> slots(routes.size() * 2 * d);
  // Each task writes disjoint [Token, slot] outputs. Scratch is thread-private.
#pragma omp parallel
  {
    std::vector<float> batch_tokens(micro_batch * d), batch_outputs(micro_batch * d), scratch;
#pragma omp for schedule(runtime)
    for (std::ptrdiff_t task_id = 0; task_id < static_cast<std::ptrdiff_t>(tasks.size());
         ++task_id) {
      const auto& task = tasks[static_cast<std::size_t>(task_id)];
      const auto& assignments = by_expert[task.expert];
      for (std::size_t item = 0; item < task.count; ++item) {
        const std::size_t token = assignments[task.begin + item] / 2;
        std::copy_n(tokens.data() + token * d, d, batch_tokens.data() + item * d);
      }
      moe::expert_forward_cpu_batch(model.experts[task.expert], config, batch_tokens.data(),
                                    task.count, batch_outputs.data(), scratch);
      for (std::size_t item = 0; item < task.count; ++item) {
        std::copy_n(batch_outputs.data() + item * d, d,
                    slots.data() + assignments[task.begin + item] * d);
      }
    }
  }
  if (timing != nullptr) timing->compute = MPI_Wtime() - compute_start;
  const double combine_start = MPI_Wtime();
  for (std::size_t token = 0; token < routes.size(); ++token) {
    for (std::size_t j = 0; j < d; ++j) {
      output[token * d + j] =
          routes[token].gate_weights[0] * slots[(token * 2) * d + j] +
          routes[token].gate_weights[1] * slots[(token * 2 + 1) * d + j];
    }
  }
  if (timing != nullptr) timing->combine = MPI_Wtime() - combine_start;
  return output;
}

std::vector<float> forward_ep(const moe::MoEConfig& config, const moe::Model& model,
                              const std::vector<float>& tokens,
                              const std::vector<moe::RoutingDecision>& routes,
                              moe::TokenRange range, int rank, int world,
                              PhaseTiming* timing = nullptr) {
  const double dispatch_pack_start = MPI_Wtime();
  Buffers dispatch(world);
  for (std::size_t local = 0; local < routes.size(); ++local) {
    const float* token = tokens.data() + local * config.dimension;
    for (std::uint8_t slot = 0; slot < 2; ++slot) {
      const auto expert = routes[local].expert_ids[slot];
      const int owner = moe::round_robin_expert_owner(expert, world);
      const auto wire = moe::pack_header(
          {range.begin + local, static_cast<std::uint32_t>(rank), expert, slot});
      append(dispatch[owner], wire, token, config.dimension);
    }
  }
  if (timing != nullptr) timing->dispatch_pack = MPI_Wtime() - dispatch_pack_start;
  const double dispatch_comm_start = MPI_Wtime();
  const auto received = exchange(dispatch, config.dimension, config.communication,
                                 1000, MPI_COMM_WORLD);
  if (timing != nullptr) timing->dispatch_comm = MPI_Wtime() - dispatch_comm_start;
  const double dispatch_check_start = MPI_Wtime();
  std::string error;
  int valid = moe::valid_assignment_buffer(received, config, world, &error) ? 1 : 0;
  if (valid) {
    for (const auto& wire : received.headers) {
      if (moe::round_robin_expert_owner(moe::unpack_header(wire).expert_id, world) != rank) {
        valid = 0;
        break;
      }
    }
  }
  int all_valid = 0;
  MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_valid) {
    throw std::runtime_error("invalid dispatched assignment: " + error);
  }
  if (timing != nullptr) timing->validation_unpack += MPI_Wtime() - dispatch_check_start;
  const double compute_start = MPI_Wtime();
  Buffers returns(world);
  std::vector<float> computed(received.headers.size() * config.dimension);
  const std::size_t d = config.dimension;
  std::vector<std::vector<std::size_t>> by_expert(config.experts);
  for (std::size_t i = 0; i < received.headers.size(); ++i) {
    by_expert[moe::unpack_header(received.headers[i]).expert_id].push_back(i);
  }
  const auto tasks = make_tasks(by_expert);
  // MPI calls stay outside OpenMP regions (MPI_THREAD_FUNNELED).
#pragma omp parallel
  {
    std::vector<float> batch_tokens(micro_batch * d), batch_outputs(micro_batch * d), scratch;
#pragma omp for schedule(runtime)
    for (std::ptrdiff_t task_id = 0; task_id < static_cast<std::ptrdiff_t>(tasks.size());
         ++task_id) {
      const auto& task = tasks[static_cast<std::size_t>(task_id)];
      const auto& assignments = by_expert[task.expert];
      for (std::size_t item = 0; item < task.count; ++item) {
        const std::size_t received_index = assignments[task.begin + item];
        std::copy_n(received.token_values.data() + received_index * d, d,
                    batch_tokens.data() + item * d);
      }
      moe::expert_forward_cpu_batch(model.experts[task.expert], config, batch_tokens.data(),
                                    task.count, batch_outputs.data(), scratch);
      for (std::size_t item = 0; item < task.count; ++item) {
        std::copy_n(batch_outputs.data() + item * d, d,
                    computed.data() + assignments[task.begin + item] * d);
      }
    }
  }
  if (timing != nullptr) timing->compute = MPI_Wtime() - compute_start;
  const double return_pack_start = MPI_Wtime();
  for (std::size_t i = 0; i < received.headers.size(); ++i) {
    const auto header = moe::unpack_header(received.headers[i]);
    append(returns[header.source_rank], received.headers[i],
           computed.data() + i * config.dimension,
           config.dimension);
  }
  if (timing != nullptr) timing->return_pack = MPI_Wtime() - return_pack_start;
  const double return_comm_start = MPI_Wtime();
  const auto returned = exchange(returns, config.dimension, config.communication,
                                 1100, MPI_COMM_WORLD);
  if (timing != nullptr) timing->return_comm = MPI_Wtime() - return_comm_start;
  const double return_unpack_start = MPI_Wtime();
  const std::size_t expected = routes.size() * 2U;
  valid = returned.headers.size() == expected;
  std::vector<unsigned char> seen(expected, 0);
  std::vector<float> slots(expected * config.dimension, 0.0F);
  for (std::size_t i = 0; i < returned.headers.size(); ++i) {
    const auto header = moe::unpack_header(returned.headers[i]);
    if (header.source_rank != static_cast<std::uint32_t>(rank) ||
        header.global_token_id < range.begin ||
        header.global_token_id >= range.begin + range.count || header.slot_id >= 2U) {
      valid = 0;
      continue;
    }
    const std::size_t local = header.global_token_id - range.begin;
    const std::size_t slot_index = local * 2U + header.slot_id;
    if (seen[slot_index] || routes[local].expert_ids[header.slot_id] != header.expert_id) {
      valid = 0;
      continue;
    }
    seen[slot_index] = 1;
    std::copy_n(returned.token_values.data() + i * config.dimension, config.dimension,
                slots.data() + slot_index * config.dimension);
  }
  for (auto item : seen) valid &= item == 1U;
  MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_valid) throw std::runtime_error("missing, duplicate, or misrouted return slot");
  if (timing != nullptr) timing->validation_unpack += MPI_Wtime() - return_unpack_start;
  const double combine_start = MPI_Wtime();
  std::vector<float> output(tokens.size());
  for (std::size_t local = 0; local < routes.size(); ++local) {
    for (std::uint32_t j = 0; j < config.dimension; ++j) {
      output[local * config.dimension + j] =
          routes[local].gate_weights[0] * slots[(local * 2U) * config.dimension + j] +
          routes[local].gate_weights[1] * slots[(local * 2U + 1U) * config.dimension + j];
    }
  }
  if (timing != nullptr) timing->combine = MPI_Wtime() - combine_start;
  return output;
}

#ifndef MOE_CPU_PILOT
int run(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  std::vector<std::string> arguments(argv + 1, argv + argc);
  const auto parsed = moe::parse_command_line(arguments);
  int valid = parsed.ok && !parsed.help_requested && provided >= MPI_THREAD_FUNNELED;
  if (valid) {
    const auto& c = parsed.config;
    valid = c.expert_backend == moe::ExpertBackend::kCpu &&
            c.placement == moe::Placement::kRoundRobin && c.tokens <= 8192 &&
            (c.parallelism == moe::Parallelism::kExpertParallel ||
             c.communication == moe::Communication::kAlltoallv);
  }
  MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!valid) {
    if (rank == 0) std::cerr << "Use valid CPU options, RR placement, N<=8192; "
                               "this program checks correctness, not performance.\n";
    MPI_Finalize();
    return 2;
  }
  try {
    const auto& config = parsed.config;
    omp_set_schedule(config.omp_schedule == moe::OpenMPSchedule::kStatic
                         ? omp_sched_static : omp_sched_dynamic, 1);
    const double setup_start = MPI_Wtime();
    const auto range = moe::contiguous_token_range(config.tokens, rank, world);
    const auto model = config.parallelism == moe::Parallelism::kExpertParallel
                           ? moe::generate_round_robin_partitioned_model(config, rank, world)
                           : moe::generate_model(config);
    const auto tokens = moe::generate_tokens(config, range.begin, range.count);
    const auto trace = config.router_mode == moe::RouterMode::kTrace
                           ? moe::generate_controlled_trace(config)
                           : std::vector<moe::RoutingDecision>{};
    const auto routes = local_routes(config, model, tokens, range, trace);
    const double setup_seconds = MPI_Wtime() - setup_start;
    MPI_Barrier(MPI_COMM_WORLD);
    const double forward_start = MPI_Wtime();
    const auto output = config.parallelism == moe::Parallelism::kDataParallel
                            ? forward_dp(config, model, tokens, routes)
                            : forward_ep(config, model, tokens, routes, range, rank, world);
    const double forward_seconds = MPI_Wtime() - forward_start;
    double setup_max = 0.0, forward_max = 0.0;
    MPI_Reduce(&setup_seconds, &setup_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&forward_seconds, &forward_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // Generate the same global input on every rank solely for this small correctness gate.
    const auto all_tokens = moe::generate_tokens(config, 0, config.tokens);
    const auto* reference_trace =
        config.router_mode == moe::RouterMode::kTrace ? &trace : nullptr;
    const auto reference = config.parallelism == moe::Parallelism::kExpertParallel
                               ? moe::forward_serial(config, moe::generate_model(config),
                                                     all_tokens, reference_trace)
                               : moe::forward_serial(config, model, all_tokens,
                                                     reference_trace);
    valid = output.size() == tokens.size();
    if (valid) {
      const float* expected = reference.output.data() + range.begin * config.dimension;
      if (!output.empty()) {
        valid = std::memcmp(output.data(), expected, output.size() * sizeof(float)) == 0;
      }
      for (const float value : output) valid &= std::isfinite(value);
    }
    int passed = 0;
    MPI_Allreduce(&valid, &passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "CPU " << moe::to_string(config.parallelism) << " correctness: "
                << (passed ? "PASS" : "FAIL") << " (P=" << world
                << ", N=" << config.tokens
                << ", communication=" << moe::to_string(config.communication) << ")\n";
      std::cout << "diagnostic_setup_seconds_max=" << setup_max << '\n'
                << "diagnostic_forward_seconds_max=" << forward_max << '\n';
    }
    MPI_Finalize();
    return passed ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "rank " << rank << ": " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
#else
int run_pilot(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  const auto parsed = moe::parse_command_line(std::vector<std::string>(argv + 1, argv + argc));
  int valid = parsed.ok && !parsed.help_requested && provided >= MPI_THREAD_FUNNELED;
  if (valid) {
    const auto& c = parsed.config;
    valid = c.expert_backend == moe::ExpertBackend::kCpu &&
            c.placement == moe::Placement::kRoundRobin &&
            (c.parallelism == moe::Parallelism::kExpertParallel ||
             c.communication == moe::Communication::kAlltoallv);
  }
  MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!valid) {
    if (rank == 0) std::cerr << "Use valid CPU options and RR placement for the pilot.\n";
    MPI_Finalize();
    return 2;
  }
  try {
    const auto& config = parsed.config;
    omp_set_schedule(config.omp_schedule == moe::OpenMPSchedule::kStatic
                         ? omp_sched_static : omp_sched_dynamic, 1);
    const auto range = moe::contiguous_token_range(config.tokens, rank, world);
    const auto model = config.parallelism == moe::Parallelism::kExpertParallel
                           ? moe::generate_round_robin_partitioned_model(config, rank, world)
                           : moe::generate_model(config);
    const auto tokens = moe::generate_tokens(config, range.begin, range.count);
    const auto trace = config.router_mode == moe::RouterMode::kTrace
                           ? moe::generate_controlled_trace(config)
                           : std::vector<moe::RoutingDecision>{};
    const auto routes = local_routes(config, model, tokens, range, trace);

    // The serial oracle runs only on rank 0 and entirely outside measurement.
    std::vector<float> reference_output;
    std::vector<int> counts, offsets;
    if (rank == 0) {
      const auto global_tokens = moe::generate_tokens(config, 0, config.tokens);
      const auto* reference_trace =
          config.router_mode == moe::RouterMode::kTrace ? &trace : nullptr;
      reference_output = config.parallelism == moe::Parallelism::kExpertParallel
                             ? moe::forward_serial(config, moe::generate_model(config),
                                                   global_tokens, reference_trace).output
                             : moe::forward_serial(config, model, global_tokens,
                                                   reference_trace).output;
      counts.resize(world);
      offsets.resize(world);
      for (int peer = 0; peer < world; ++peer) {
        const auto peer_range = moe::contiguous_token_range(config.tokens, peer, world);
        counts[peer] = mpi_count(peer_range.count * config.dimension);
        offsets[peer] = mpi_count(peer_range.begin * config.dimension);
      }
    }
    std::vector<float> expected(tokens.size());
    float empty_receive = 0.0F;
    MPI_Scatterv(rank == 0 ? reference_output.data() : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? offsets.data() : nullptr, MPI_FLOAT,
                 expected.empty() ? &empty_receive : expected.data(),
                 mpi_count(expected.size()),
                 MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto forward = [&](PhaseTiming* timing) {
      return config.parallelism == moe::Parallelism::kDataParallel
                 ? forward_dp(config, model, tokens, routes, timing)
                 : forward_ep(config, model, tokens, routes, range, rank, world, timing);
    };
    auto check_output = [&](const std::vector<float>& output) {
      int local_ok = output.size() == expected.size();
      if (local_ok && !output.empty()) {
        local_ok = std::memcmp(output.data(), expected.data(),
                               output.size() * sizeof(float)) == 0;
      }
      for (const float value : output) local_ok &= std::isfinite(value);
      int all_ok = 0;
      MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      return all_ok != 0;
    };

    constexpr int warmups = 3;
    constexpr int repeats = 5;
    for (int i = 0; i < warmups; ++i) {
      if (!check_output(forward(nullptr))) {
        if (rank == 0) std::cerr << "pilot warmup correctness: FAIL\n";
        MPI_Finalize();
        return 1;
      }
    }
    for (int trial = 0; trial < repeats; ++trial) {
      MPI_Barrier(MPI_COMM_WORLD);
      const double start = MPI_Wtime();
      PhaseTiming phases;
      const auto output = forward(&phases);
      const double local_seconds = MPI_Wtime() - start;
      double max_seconds = 0.0;
      MPI_Reduce(&local_seconds, &max_seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      const std::array<double, 7> local_phases{
          phases.dispatch_pack, phases.dispatch_comm, phases.compute,
          phases.return_pack, phases.return_comm, phases.validation_unpack,
          phases.combine};
      std::array<double, 7> max_phases{};
      MPI_Reduce(local_phases.data(), max_phases.data(),
                 static_cast<int>(local_phases.size()), MPI_DOUBLE, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      const bool passed = check_output(output);
      if (rank == 0) {
        std::cout << "pilot,trial=" << trial << ",parallelism="
                  << moe::to_string(config.parallelism) << ",communication="
                  << moe::to_string(config.communication) << ",tokens=" << config.tokens
                  << ",experts=" << config.experts << ",ranks=" << world
                  << ",threads=" << omp_get_max_threads()
                  << ",end_to_end_seconds=" << max_seconds
                  << ",dispatch_pack_seconds=" << max_phases[0]
                  << ",dispatch_comm_seconds=" << max_phases[1]
                  << ",expert_compute_seconds=" << max_phases[2]
                  << ",return_pack_seconds=" << max_phases[3]
                  << ",return_comm_seconds=" << max_phases[4]
                  << ",validation_unpack_seconds=" << max_phases[5]
                  << ",combine_seconds=" << max_phases[6]
                  << ",bitwise_ok=" << (passed ? 1 : 0) << '\n';
      }
      if (!passed) {
        MPI_Finalize();
        return 1;
      }
    }
    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef MOE_CPU_PILOT
  return run_pilot(argc, argv);
#else
  return run(argc, argv);
#endif
}
