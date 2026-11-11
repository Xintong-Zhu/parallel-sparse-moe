#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

namespace {

struct Options {
  int expected_threads = -1;
  bool require_distinct_hosts = false;
  bool check_affinity = false;
  bool require_hybrid_layout = false;
};

const char* thread_level_name(int level) {
  switch (level) {
    case MPI_THREAD_SINGLE:
      return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED:
      return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED:
      return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE:
      return "MPI_THREAD_MULTIPLE";
    default:
      return "unknown";
  }
}

bool parse_options(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--require-distinct-hosts") {
      options.require_distinct_hosts = true;
    } else if (argument == "--check-affinity") {
      options.check_affinity = true;
    } else if (argument == "--require-hybrid-layout") {
      options.check_affinity = true;
      options.require_hybrid_layout = true;
    } else if (argument == "--expect-threads" && i + 1 < argc) {
      try {
        options.expected_threads = std::stoi(argv[++i]);
      } catch (const std::exception&) {
        error = "--expect-threads requires a positive integer";
        return false;
      }
      if (options.expected_threads <= 0) {
        error = "--expect-threads requires a positive integer";
        return false;
      }
    } else {
      error = "unknown or incomplete option: " + argument;
      return false;
    }
  }
  return !options.check_affinity || options.expected_threads > 0;
}

// Each OpenMP thread must be restricted to a single physical core. SMT sibling
// masks are allowed, but a physical core cannot be shared by different threads.
std::array<int, 4> thread_affinity() {
  std::array<int, 4> record{-1, -1, -1, 0}; // current CPU, package, core, valid
#ifdef __linux__
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return record;
  record[0] = sched_getcpu();
  bool found = false;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) continue;
    const std::string prefix = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    int package = -1, core = -1;
    if (!(std::ifstream(prefix + "physical_package_id") >> package) ||
        !(std::ifstream(prefix + "core_id") >> core)) return record;
    if (found && (package != record[1] || core != record[2])) return record;
    record[1] = package;
    record[2] = core;
    found = true;
  }
  record[3] = found && record[0] >= 0 && record[0] < CPU_SETSIZE && CPU_ISSET(record[0], &mask);
#endif
  return record;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  Options options;
  std::string option_error;
  const bool valid_options = parse_options(argc, argv, options, option_error);
  int valid_options_int = valid_options ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &valid_options_int, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if (!valid_options_int) {
    if (rank == 0) {
      std::cerr << "Usage: moe_launch_probe [--expect-threads N] "
                   "[--require-distinct-hosts] [--check-affinity] [--require-hybrid-layout]\n";
      if (!option_error.empty()) {
        std::cerr << "Error: " << option_error << '\n';
      }
    }
    MPI_Finalize();
    return 2;
  }

  if (provided < MPI_THREAD_FUNNELED) {
    if (rank == 0) {
      std::cerr << "MPI implementation provided " << thread_level_name(provided)
                << "; this project requires at least MPI_THREAD_FUNNELED.\n";
    }
    MPI_Finalize();
    return 3;
  }

  std::array<char, MPI_MAX_PROCESSOR_NAME> hostname{};
  int hostname_length = 0;
  MPI_Get_processor_name(hostname.data(), &hostname_length);
  hostname[std::min(hostname_length, MPI_MAX_PROCESSOR_NAME - 1)] = '\0';

  int observed_threads = 0;
  std::vector<int> affinity(options.check_affinity ? options.expected_threads * 4 : 0, -1);
#pragma omp parallel
  {
#pragma omp single
    observed_threads = omp_get_num_threads();
    if (options.check_affinity && omp_get_thread_num() < options.expected_threads) {
      const auto record = thread_affinity();
      std::copy(record.begin(), record.end(), affinity.begin() + omp_get_thread_num() * 4);
    }
  }

  const int successor = (rank + 1) % world_size;
  const int predecessor = (rank - 1 + world_size) % world_size;
  const int sent_value = rank;
  int received_value = -1;
  MPI_Sendrecv(&sent_value, 1, MPI_INT, successor, 101, &received_value, 1,
               MPI_INT, predecessor, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  int local_ok = (received_value == predecessor) ? 1 : 0;

  std::vector<int> all_threads;
  std::vector<int> all_ring_ok;
  std::vector<char> all_hostnames;
  if (rank == 0) {
    all_threads.resize(world_size);
    all_ring_ok.resize(world_size);
    all_hostnames.resize(static_cast<std::size_t>(world_size) *
                         MPI_MAX_PROCESSOR_NAME);
  }

  MPI_Gather(&observed_threads, 1, MPI_INT, all_threads.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&local_ok, 1, MPI_INT, all_ring_ok.data(), 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(hostname.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
             all_hostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0,
             MPI_COMM_WORLD);

  std::vector<int> all_affinity(rank == 0 ? affinity.size() * world_size : 0);
  if (options.check_affinity) {
    MPI_Gather(affinity.data(), static_cast<int>(affinity.size()), MPI_INT,
               all_affinity.data(), static_cast<int>(affinity.size()), MPI_INT, 0, MPI_COMM_WORLD);
  }

  int passed = 1;
  if (rank == 0) {
    std::set<std::string> hosts;
    std::cout << "MPI/OpenMP launch probe\n"
              << "  MPI ranks: " << world_size << '\n'
              << "  MPI thread support: " << thread_level_name(provided) << '\n'
              << "  OMP_NUM_THREADS: "
              << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS")
                                                   : "UNSET")
              << "\n";
    for (int current_rank = 0; current_rank < world_size; ++current_rank) {
      const char* current_host =
          all_hostnames.data() + current_rank * MPI_MAX_PROCESSOR_NAME;
      hosts.emplace(current_host);
      std::cout << "  rank " << current_rank << ": host=" << current_host
                << ", OpenMP threads=" << all_threads[current_rank]
                << ", ring=" << (all_ring_ok[current_rank] ? "ok" : "failed")
                << '\n';
      if (!all_ring_ok[current_rank]) {
        passed = 0;
      }
      if (options.expected_threads > 0 &&
          all_threads[current_rank] != options.expected_threads) {
        passed = 0;
      }
    }
    if (options.require_distinct_hosts && hosts.size() < 2) {
      std::cerr << "Expected ranks on at least two hosts, but observed "
                << hosts.size() << ".\n";
      passed = 0;
    }
    if (options.check_affinity) {
      std::set<std::string> physical_cores;
      std::map<std::string, int> host_counts, socket_counts;
      for (int peer = 0; peer < world_size; ++peer) {
        const std::string host = all_hostnames.data() + peer * MPI_MAX_PROCESSOR_NAME;
        int rank_package = -1;
        for (int tid = 0; tid < options.expected_threads; ++tid) {
          const auto offset = (peer * options.expected_threads + tid) * 4;
          const int cpu = all_affinity[offset], package = all_affinity[offset + 1];
          const int core = all_affinity[offset + 2], valid = all_affinity[offset + 3];
          const std::string socket = host + "," + std::to_string(package);
          const std::string key = socket + "," + std::to_string(core);
          if (valid != 1 || !physical_cores.insert(key).second ||
              (tid > 0 && package != rank_package)) passed = 0;
          rank_package = package;
          ++host_counts[host];
          ++socket_counts[socket];
          std::cout << "affinity,rank=" << peer << ",thread=" << tid << ",host=" << host
                    << ",cpu=" << cpu << ",package=" << package << ",core=" << core
                    << ",valid=" << valid << '\n';
        }
      }
      if (options.require_hybrid_layout) {
        if (physical_cores.size() != 256 || host_counts.size() != 4 || socket_counts.size() != 8) passed = 0;
        for (const auto& entry : host_counts) if (entry.second != 64) passed = 0;
        for (const auto& entry : socket_counts) if (entry.second != 32) passed = 0;
      }
      for (const auto& core : physical_cores) std::cout << "physical_core," << core << '\n';
      std::cout << "Affinity check: " << (passed ? "PASS" : "FAIL") << '\n';
    }
    if (passed) {
      std::cout << "RESULT: PASS\n";
    } else {
      std::cerr << "RESULT: FAIL\n";
    }
  }

  MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return passed ? 0 : 4;
}
