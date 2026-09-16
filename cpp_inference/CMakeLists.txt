cmake_minimum_required(VERSION 3.20)

project(resnet18_ort_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.tools/onnxruntime-win-x64-1.23.2" CACHE PATH
    "Path to the project-local ONNX Runtime SDK")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ORT SDK was not found at: ${ORT_ROOT}")
endif()

if(NOT EXISTS "${ORT_ROOT}/lib/onnxruntime.dll")
    message(FATAL_ERROR "ORT runtime DLL was not found at: ${ORT_ROOT}/lib/onnxruntime.dll")
endif()

add_executable(resnet18_ort_cpp src/main.cpp)
target_compile_options(resnet18_ort_cpp PRIVATE -std=gnu++17 -Wall -Wextra -Wpedantic)
target_include_directories(resnet18_ort_cpp PRIVATE "${ORT_ROOT}/include")
target_link_directories(resnet18_ort_cpp PRIVATE "${ORT_ROOT}/lib")

# windowscodecs is the Windows Imaging Component API used for in-process JPEG decoding.
target_link_libraries(resnet18_ort_cpp PRIVATE onnxruntime windowscodecs ole32 uuid)

# Prevent accidental loading of a system-wide older onnxruntime.dll at runtime.
add_custom_command(TARGET resnet18_ort_cpp POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:resnet18_ort_cpp>/onnxruntime.dll"
    COMMENT "Copying ONNX Runtime 1.23.2 DLL next to the executable")
