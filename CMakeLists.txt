cmake_minimum_required(VERSION 3.18)
project(fp8_gemm_jit LANGUAGES CXX CUDA)

# Set C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA components
find_package(CUDA 11.8 REQUIRED)
find_library(NVRTC_LIB nvrtc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDA_DRIVER_LIB cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Find rapidhash
find_path(
  RAPIDHASH_INCLUDE_DIR rapidhash.h # 指定要查找的文件名
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidhash" # 指定搜索路径
  NO_DEFAULT_PATH # 禁止搜索系统路径
)
if(NOT RAPIDHASH_INCLUDE_DIR)
  message(FATAL_ERROR "rapidhash.h not found in third_party/rapidhash")
endif()

# Include directories
include_directories(include ${RAPIDHASH_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# Source files
file(GLOB SOURCES "src/*.cpp")

# Main application
add_executable(fp8_gemm_demo main.cpp ${SOURCES})
target_link_libraries(fp8_gemm_demo ${NVRTC_LIB} ${CUDA_DRIVER_LIB}
                      ${CUDA_LIBRARIES} cudart)

# Test executables
add_executable(test_template tests/test_template.cpp ${SOURCES})
target_link_libraries(test_template ${NVRTC_LIB} ${CUDA_DRIVER_LIB}
                      ${CUDA_LIBRARIES} cudart)

add_executable(test_jit tests/test_jit.cpp ${SOURCES})
target_link_libraries(test_jit ${NVRTC_LIB} ${CUDA_DRIVER_LIB}
                      ${CUDA_LIBRARIES} cudart)

add_executable(test_runtime tests/test_runtime.cpp ${SOURCES})
target_link_libraries(test_runtime ${NVRTC_LIB} ${CUDA_DRIVER_LIB}
                      ${CUDA_LIBRARIES} cudart)

# Compiler options - optimized for H100
set_target_properties(
  fp8_gemm_demo PROPERTIES CUDA_ARCHITECTURES "90" # H100 architecture (sm_90)
)

set_target_properties(test_template PROPERTIES CUDA_ARCHITECTURES "90")

set_target_properties(test_jit PROPERTIES CUDA_ARCHITECTURES "90")

set_target_properties(test_runtime PROPERTIES CUDA_ARCHITECTURES "90")

# Additional optimizations
target_compile_options(
  fp8_gemm_demo
  PRIVATE -O3 -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
          $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

# Test compile options
foreach(test_target test_template test_jit test_runtime)
  target_compile_options(
    ${test_target} PRIVATE -O3 -Wall -Wextra
                           $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endforeach()

# Install targets
install(TARGETS fp8_gemm_demo test_template test_jit test_runtime
        RUNTIME DESTINATION bin)
