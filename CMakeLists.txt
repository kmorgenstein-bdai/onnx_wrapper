cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference_test)

set(TORCH_LIBRARIES "${PROJECT_SOURCE_DIR}/external/libtorch")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_LIBRARIES}")
find_package(Torch REQUIRED)

set(ONNX_LIBRARIES "${PROJECT_SOURCE_DIR}/external/onnxruntime-linux-x64-1.15.1")
include_directories(
    "${ONNX_LIBRARIES}/include"
    "${ONNX_LIBRARIES}/include/onnxruntime"
    )
link_directories("${ONNX_LIBRARIES}/lib")



string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test test.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}" onnxruntime)
set_property(TARGET test PROPERTY CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/external/ParamHandler)

add_executable(main main.cpp)
target_link_libraries(main onnxruntime)
set_property(TARGET main PROPERTY CXX_STANDARD 17)