cmake_minimum_required(VERSION 3.20)

project(parallel_sparse_moe LANGUAGES CXX)

include(CTest)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
option(MOE_ENABLE_CUDA "Build the optional single-GPU Expert benchmark" OFF)

add_library(moe_core STATIC
  src/config.cpp
  src/data.cpp
  src/expert.cpp
  src/placement.cpp
  src/protocol.cpp
  src/router.cpp
  src/serial.cpp
)
target_compile_features(moe_core PUBLIC cxx_std_17)
target_include_directories(moe_core PUBLIC include)

add_executable(moe_launch_probe src/launch_probe.cpp)
target_compile_features(moe_launch_probe PRIVATE cxx_std_17)
target_link_libraries(moe_launch_probe PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(moe_contract_check src/contract_check.cpp)
target_link_libraries(moe_contract_check PRIVATE moe_core)

add_executable(moe_serial src/serial_main.cpp)
target_link_libraries(moe_serial PRIVATE moe_core)

add_executable(moe_serial_bench src/serial_bench.cpp)
target_link_libraries(moe_serial_bench PRIVATE moe_core)

add_executable(moe_mpi_loopback src/mpi_loopback.cpp)
target_link_libraries(moe_mpi_loopback PRIVATE moe_core MPI::MPI_CXX)

add_executable(moe_parallel_check src/parallel_check.cpp)
target_link_libraries(moe_parallel_check PRIVATE moe_core MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(moe_cpu_pilot src/parallel_check.cpp)
target_compile_definitions(moe_cpu_pilot PRIVATE MOE_CPU_PILOT=1)
target_link_libraries(moe_cpu_pilot PRIVATE moe_core MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(moe_cpu_bench src/parallel_check.cpp)
target_compile_definitions(moe_cpu_bench PRIVATE MOE_CPU_BENCH=1)
target_link_libraries(moe_cpu_bench PRIVATE moe_core MPI::MPI_CXX OpenMP::OpenMP_CXX)

if(MOE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(moe_gpu src/gpu_main.cu)
  target_link_libraries(moe_gpu PRIVATE moe_core CUDA::cudart)
  set_target_properties(moe_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                        CUDA_ARCHITECTURES 86)
  target_compile_options(moe_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

if(BUILD_TESTING)
  add_executable(moe_protocol_tests tests/protocol_tests.cpp)
  target_link_libraries(moe_protocol_tests PRIVATE moe_core)
  add_test(NAME moe_protocol_tests COMMAND moe_protocol_tests)

  add_executable(moe_serial_tests tests/serial_tests.cpp)
  target_link_libraries(moe_serial_tests PRIVATE moe_core)
  add_test(NAME moe_serial_tests COMMAND moe_serial_tests)

  add_executable(moe_placement_tests tests/placement_tests.cpp)
  target_link_libraries(moe_placement_tests PRIVATE moe_core)
  add_test(NAME moe_placement_tests COMMAND moe_placement_tests)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(moe_core PRIVATE -Wall -Wextra -Wpedantic)
  # Callers also perform FP32 slot combination, including the test oracle and
  # timed benchmark. All C++ consumers must use the same rounding policy.
  # CUDA has its own --fmad=false option above.
  target_compile_options(moe_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  )
  target_compile_options(moe_launch_probe PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_contract_check PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_serial PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_serial_bench PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_mpi_loopback PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_parallel_check PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_cpu_pilot PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(moe_cpu_bench PRIVATE -Wall -Wextra -Wpedantic)
  if(BUILD_TESTING)
    target_compile_options(moe_protocol_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(moe_serial_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(moe_placement_tests PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
