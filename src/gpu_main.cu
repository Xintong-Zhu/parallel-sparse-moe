#include "moe/config.hpp"
#include "moe/data.hpp"
#include "moe/router.hpp"
#include "moe/serial.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

double seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

__global__ void first_layer(const float* inputs, const std::uint32_t* expert_ids,
                            const float* w1, float* hidden, std::size_t assignments,
                            std::uint32_t d, std::uint32_t h) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= assignments * h) return;
  const std::size_t assignment = index / h;
  const std::size_t hidden_feature = index % h;
  const std::size_t expert = expert_ids[assignment];
  float value = 0.0F;
  for (std::uint32_t feature = 0; feature < d; ++feature) {
    value += inputs[assignment * d + feature] *
             w1[(expert * d + feature) * h + hidden_feature];
  }
  hidden[index] = value > 0.0F ? value : 0.0F;
}

__global__ void second_layer(const float* hidden, const std::uint32_t* expert_ids,
                             const float* w2, float* outputs, std::size_t assignments,
                             std::uint32_t d, std::uint32_t h) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= assignments * d) return;
  const std::size_t assignment = index / d;
  const std::size_t feature = index % d;
  const std::size_t expert = expert_ids[assignment];
  float value = 0.0F;
  for (std::uint32_t hidden_feature = 0; hidden_feature < h; ++hidden_feature) {
    value += hidden[assignment * h + hidden_feature] *
             w2[(expert * h + hidden_feature) * d + feature];
  }
  outputs[index] = value;
}

struct GpuTimes {
  double h2d = 0.0;
  double kernel = 0.0;
  double d2h = 0.0;
};

class GpuExperts {
 public:
  GpuExperts(const moe::MoEConfig& config, const moe::Model& model)
      : d_(config.dimension), h_(config.hidden_dimension),
        assignments_(static_cast<std::size_t>(config.tokens) * 2U) {
    const std::size_t w1_count = static_cast<std::size_t>(config.experts) * d_ * h_;
    const std::size_t w2_count = static_cast<std::size_t>(config.experts) * h_ * d_;
    std::vector<float> w1(w1_count), w2(w2_count);
    for (std::size_t expert = 0; expert < config.experts; ++expert) {
      std::copy(model.experts[expert].w1.begin(), model.experts[expert].w1.end(),
                w1.begin() + expert * d_ * h_);
      std::copy(model.experts[expert].w2.begin(), model.experts[expert].w2.end(),
                w2.begin() + expert * h_ * d_);
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_w1_), w1_count * sizeof(float)),
               "cudaMalloc W1");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_w2_), w2_count * sizeof(float)),
               "cudaMalloc W2");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_inputs_), assignments_ * d_ * sizeof(float)),
               "cudaMalloc inputs");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_ids_), assignments_ * sizeof(std::uint32_t)),
               "cudaMalloc Expert IDs");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_hidden_), assignments_ * h_ * sizeof(float)),
               "cudaMalloc hidden");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_outputs_), assignments_ * d_ * sizeof(float)),
               "cudaMalloc outputs");
    const auto start = Clock::now();
    cuda_check(cudaMemcpy(device_w1_, w1.data(), w1_count * sizeof(float), cudaMemcpyHostToDevice),
               "upload W1");
    cuda_check(cudaMemcpy(device_w2_, w2.data(), w2_count * sizeof(float), cudaMemcpyHostToDevice),
               "upload W2");
    weight_h2d_seconds_ = seconds(start, Clock::now());
    cuda_check(cudaEventCreate(&kernel_start_), "create kernel start event");
    cuda_check(cudaEventCreate(&kernel_end_), "create kernel end event");
  }

  GpuExperts(const GpuExperts&) = delete;
  GpuExperts& operator=(const GpuExperts&) = delete;

  ~GpuExperts() {
    if (kernel_start_) cudaEventDestroy(kernel_start_);
    if (kernel_end_) cudaEventDestroy(kernel_end_);
    if (device_w1_) cudaFree(device_w1_);
    if (device_w2_) cudaFree(device_w2_);
    if (device_inputs_) cudaFree(device_inputs_);
    if (device_ids_) cudaFree(device_ids_);
    if (device_hidden_) cudaFree(device_hidden_);
    if (device_outputs_) cudaFree(device_outputs_);
  }

  double weight_h2d_seconds() const { return weight_h2d_seconds_; }

  GpuTimes forward(const std::vector<float>& inputs,
                   const std::vector<std::uint32_t>& expert_ids,
                   std::vector<float>& outputs) {
    if (inputs.size() != assignments_ * d_ || expert_ids.size() != assignments_ ||
        outputs.size() != inputs.size()) {
      throw std::invalid_argument("GPU assignment buffer shape mismatch");
    }
    GpuTimes result;
    auto start = Clock::now();
    cuda_check(cudaMemcpy(device_inputs_, inputs.data(), inputs.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "upload assignments");
    cuda_check(cudaMemcpy(device_ids_, expert_ids.data(), expert_ids.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "upload Expert IDs");
    result.h2d = seconds(start, Clock::now());

    constexpr unsigned int threads = 256;
    const auto first_count = assignments_ * h_;
    const auto second_count = assignments_ * d_;
    const auto first_blocks = static_cast<unsigned int>((first_count + threads - 1U) / threads);
    const auto second_blocks = static_cast<unsigned int>((second_count + threads - 1U) / threads);
    cuda_check(cudaEventRecord(kernel_start_), "record kernel start");
    first_layer<<<first_blocks, threads>>>(device_inputs_, device_ids_, device_w1_,
                                            device_hidden_, assignments_,
                                            static_cast<std::uint32_t>(d_),
                                            static_cast<std::uint32_t>(h_));
    cuda_check(cudaGetLastError(), "launch first layer");
    second_layer<<<second_blocks, threads>>>(device_hidden_, device_ids_, device_w2_,
                                              device_outputs_, assignments_,
                                              static_cast<std::uint32_t>(d_),
                                              static_cast<std::uint32_t>(h_));
    cuda_check(cudaGetLastError(), "launch second layer");
    cuda_check(cudaEventRecord(kernel_end_), "record kernel end");
    cuda_check(cudaEventSynchronize(kernel_end_), "synchronize kernels");
    float kernel_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&kernel_ms, kernel_start_, kernel_end_),
               "measure kernels");
    result.kernel = static_cast<double>(kernel_ms) / 1000.0;

    start = Clock::now();
    cuda_check(cudaMemcpy(outputs.data(), device_outputs_, outputs.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "download outputs");
    result.d2h = seconds(start, Clock::now());
    return result;
  }

 private:
  std::size_t d_, h_, assignments_;
  float *device_w1_ = nullptr, *device_w2_ = nullptr;
  float *device_inputs_ = nullptr, *device_hidden_ = nullptr, *device_outputs_ = nullptr;
  std::uint32_t* device_ids_ = nullptr;
  cudaEvent_t kernel_start_ = nullptr, kernel_end_ = nullptr;
  double weight_h2d_seconds_ = 0.0;
};

int run(int argc, char** argv) {
  const auto parsed = moe::parse_command_line(std::vector<std::string>(argv + 1, argv + argc));
  if (!parsed.ok || parsed.help_requested ||
      parsed.config.expert_backend != moe::ExpertBackend::kCuda) {
    std::cerr << "Use valid options and --backend cuda for moe_gpu.\n";
    return 2;
  }
  const auto& config = parsed.config;
  if (config.tokens > 32768U) {
    std::cerr << "moe_gpu currently supports at most 32768 Tokens.\n";
    return 2;
  }
  int devices = 0;
  cuda_check(cudaGetDeviceCount(&devices), "query GPU count");
  if (devices < 1) throw std::runtime_error("no CUDA GPU is visible");
  cuda_check(cudaSetDevice(0), "select GPU 0");
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, 0), "query GPU properties");
  int runtime_version = 0;
  cuda_check(cudaRuntimeGetVersion(&runtime_version), "query CUDA runtime");

  const auto model = moe::generate_model(config);
  const auto tokens = moe::generate_tokens(config, 0, config.tokens);
  const auto trace = config.router_mode == moe::RouterMode::kTrace
                         ? moe::generate_controlled_trace(config)
                         : std::vector<moe::RoutingDecision>{};
  const auto* trace_pointer = config.router_mode == moe::RouterMode::kTrace ? &trace : nullptr;
  const auto reference = moe::forward_serial(config, model, tokens, trace_pointer);
  const std::size_t n = static_cast<std::size_t>(config.tokens);
  const std::size_t d = config.dimension;
  std::vector<moe::RoutingDecision> routes(n);
  std::vector<float> assignment_inputs(n * 2U * d), assignment_outputs(n * 2U * d);
  std::vector<std::uint32_t> assignment_experts(n * 2U);
  std::vector<float> combined(n * d);
  GpuExperts engine(config, model);

  std::cout << "gpu_device=" << properties.name << ",visible_devices=" << devices
            << ",cuda_runtime=" << runtime_version
            << ",weights_h2d_seconds=" << engine.weight_h2d_seconds() << '\n';
  const auto histogram = moe::routing_histogram(reference.routes, config.experts);
  std::cout << "expert_histogram=";
  for (std::size_t expert = 0; expert < histogram.size(); ++expert) {
    std::cout << (expert == 0 ? "" : ",") << histogram[expert];
  }
  std::cout << '\n';
  constexpr int warmups = 3;
  constexpr int repeats = 5;
  for (int trial = -warmups; trial < repeats; ++trial) {
    const auto total_start = Clock::now();
    for (std::size_t token = 0; token < n; ++token) {
      const float* input = tokens.data() + token * d;
      routes[token] = trace_pointer ? (*trace_pointer)[token]
                                    : moe::route_token(model, config, input);
      for (std::size_t slot = 0; slot < 2; ++slot) {
        const std::size_t assignment = token * 2U + slot;
        assignment_experts[assignment] = routes[token].expert_ids[slot];
        std::copy_n(input, d, assignment_inputs.data() + assignment * d);
      }
    }
    const auto packed = Clock::now();
    const auto gpu_times = engine.forward(assignment_inputs, assignment_experts,
                                          assignment_outputs);
    const auto downloaded = Clock::now();
    for (std::size_t token = 0; token < n; ++token) {
      for (std::size_t feature = 0; feature < d; ++feature) {
        combined[token * d + feature] =
            routes[token].gate_weights[0] * assignment_outputs[(token * 2U) * d + feature] +
            routes[token].gate_weights[1] * assignment_outputs[(token * 2U + 1U) * d + feature];
      }
    }
    const auto total_end = Clock::now();

    bool routes_match = routes.size() == reference.routes.size();
    for (std::size_t token = 0; token < n && routes_match; ++token) {
      routes_match = routes[token].expert_ids == reference.routes[token].expert_ids &&
                     routes[token].gate_weights == reference.routes[token].gate_weights;
    }
    float max_abs_error = 0.0F;
    double squared_error = 0.0, squared_reference = 0.0;
    bool elementwise_ok = true;
    for (std::size_t i = 0; i < combined.size(); ++i) {
      const double actual = combined[i], expected = reference.output[i];
      const double difference = actual - expected;
      max_abs_error = std::max(max_abs_error, static_cast<float>(std::abs(difference)));
      squared_error += difference * difference;
      squared_reference += expected * expected;
      elementwise_ok &= std::isfinite(actual) &&
                        std::abs(difference) <= 1.0e-4 + 1.0e-4 * std::abs(expected);
    }
    const double relative_l2 =
        std::sqrt(squared_error / std::max(squared_reference,
                                          std::numeric_limits<double>::min()));
    const bool passed = routes_match && elementwise_ok && relative_l2 <= 1.0e-5;
    if (!passed) {
      std::cerr << "GPU correctness: FAIL; max_abs_error=" << max_abs_error
                << ", relative_l2=" << relative_l2 << '\n';
      return 1;
    }
    if (trial >= 0) {
      std::cout << "gpu_trial=" << trial << ",tokens=" << n
                << ",experts=" << config.experts
                << ",router_mode=" << moe::to_string(config.router_mode)
                << ",host_pack_seconds=" << seconds(total_start, packed)
                << ",h2d_seconds=" << gpu_times.h2d
                << ",kernel_seconds=" << gpu_times.kernel
                << ",d2h_seconds=" << gpu_times.d2h
                << ",host_combine_seconds=" << seconds(downloaded, total_end)
                << ",end_to_end_seconds=" << seconds(total_start, total_end)
                << ",max_abs_error=" << max_abs_error
                << ",relative_l2_error=" << relative_l2
                << ",correctness=PASS\n";
    }
  }
  for (int warmup = 0; warmup < warmups; ++warmup) {
    const auto cpu = moe::forward_serial(config, model, tokens, trace_pointer);
    if (cpu.output != reference.output) {
      std::cerr << "CPU warmup correctness: FAIL\n";
      return 1;
    }
  }
  for (int trial = 0; trial < repeats; ++trial) {
    const auto start = Clock::now();
    const auto cpu = moe::forward_serial(config, model, tokens, trace_pointer);
    const auto end = Clock::now();
    if (cpu.output != reference.output) {
      std::cerr << "CPU repeat correctness: FAIL\n";
      return 1;
    }
    std::cout << "cpu_trial=" << trial << ",tokens=" << n
              << ",serial_seconds=" << seconds(start, end)
              << ",correctness=PASS\n";
  }
  std::cout << "GPU correctness: PASS\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "GPU error: " << error.what() << '\n';
    return 1;
  }
}
