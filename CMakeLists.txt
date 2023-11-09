cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(onnx_wrapper)

set(ONNX_LIBRARIES "${PROJECT_SOURCE_DIR}/external/onnxruntime-linux-x64-1.15.1")
include_directories(
    "${ONNX_LIBRARIES}/include"
    "${ONNX_LIBRARIES}/include/onnxruntime"
    )
link_directories("${ONNX_LIBRARIES}/lib")

string(APPEND CMAKE_CXX_FLAGS "-Wall -Wextra")
string(APPEND CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/external)
#include_directories(${PROJECT_SOURCE_DIR}/external/ParamHandler)

add_subdirectory(external)

#target_link_libraries(dynacore_param_handler)

add_executable(main src/main.cpp)
target_link_libraries(main onnxruntime)
set_property(TARGET main PROPERTY CXX_STANDARD 17)