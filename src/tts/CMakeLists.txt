# CosyVoice2测试程序独立编译

cmake_minimum_required(VERSION 3.18)
project(CosyVoice2Test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找ONNX Runtime
if(EXISTS /usr/local/include/onnxruntime)
    set(ONNXRUNTIME_INCLUDE_DIRS /usr/local/include/onnxruntime)
    set(ONNXRUNTIME_LIB_DIRS /usr/local/lib)
else()
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime root directory")
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include/onnxruntime)
    set(ONNXRUNTIME_LIB_DIRS ${ONNXRUNTIME_ROOT}/lib)
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
link_directories(${ONNXRUNTIME_LIB_DIRS})

# 包含项目源码目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# 源文件
set(COSYVOICE2_SOURCES
    cosyvoice2_engine.cpp
    ../utils/logger.cpp
    ../utils/config.cpp
)

set(TEST_SOURCES
    test_cosyvoice2.cpp
)

# 测试可执行文件
add_executable(test_cosyvoice2
    ${TEST_SOURCES}
    ${COSYVOICE2_SOURCES}
)

# 链接库
target_link_libraries(test_cosyvoice2
    PRIVATE
        onnxruntime
)

# 编译选项
target_compile_options(test_cosyvoice2 PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0>
)

# 打印配置信息
message(STATUS "========== CosyVoice2 Test Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIRS}")
message(STATUS "===================================================")
