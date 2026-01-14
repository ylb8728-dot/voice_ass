cmake_minimum_required(VERSION 3.18)
project(VoiceAssistant VERSION 1.0.0 LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# 查找线程库
find_package(Threads REQUIRED)

# ===== ONNX Runtime ===== (Removed - replaced by whisper.cpp)
if(EXISTS /usr/local/include/onnxruntime)
    set(ONNXRUNTIME_INCLUDE_DIRS /usr/local/include/onnxruntime)
    set(ONNXRUNTIME_LIB_DIRS /usr/local/lib)
    message(STATUS "Found ONNX Runtime in system directories")
else()
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime root directory")
    if(NOT EXISTS ${ONNXRUNTIME_ROOT})
        message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
    endif()
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include/onnxruntime)
    set(ONNXRUNTIME_LIB_DIRS ${ONNXRUNTIME_ROOT}/lib)
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
link_directories(${ONNXRUNTIME_LIB_DIRS})

# ===== PortAudio =====
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

if(NOT PORTAUDIO_FOUND)
    message(FATAL_ERROR "PortAudio not found. Install with: sudo apt-get install portaudio19-dev")
endif()

# ===== SQLite3 =====
find_package(SQLite3 REQUIRED)

# ===== llama.cpp =====
set(GGML_CUDA ON CACHE BOOL "Enable CUDA support for llama.cpp")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
add_subdirectory(llama.cpp)

# ===== whisper.cpp =====
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "Build whisper tests")
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "Build whisper examples")
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "Build whisper server")
set(GGML_CUDA ON CACHE BOOL "Enable CUDA for whisper.cpp")
add_subdirectory(whisper.cpp)

# ===== CUDA (可选) =====
option(USE_CUDA "Enable CUDA support" ON)
if(USE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA not found, building CPU-only version")
        set(USE_CUDA OFF)
    endif()
endif()

# ===== yaml-cpp (配置文件解析) =====
find_package(PkgConfig)
pkg_check_modules(YAML_CPP yaml-cpp)
if(NOT YAML_CPP_FOUND)
    message(WARNING "yaml-cpp not found, install with: sudo apt-get install libyaml-cpp-dev")
endif()

# ===== spdlog (日志库) =====
find_package(spdlog)
if(NOT spdlog_FOUND)
    message(WARNING "spdlog not found, install with: sudo apt-get install libspdlog-dev")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/include
    # ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    ${PORTAUDIO_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# 源文件
set(SOURCES
    src/main.cpp
    src/audio/audio_input.cpp
    src/audio/audio_output.cpp
    src/audio/vad_detector.cpp
    src/asr/asr_engine.cpp
    # src/asr/feature_extractor.cpp  # Removed - whisper.cpp handles feature extraction
    src/query/query_engine.cpp
    src/query/sqlite_db.cpp
    src/query/llm_client.cpp
    src/tts/tts_engine.cpp
    src/tts/text_frontend.cpp
    src/utils/logger.cpp
    src/utils/config.cpp
    src/utils/thread_pool.cpp
    src/utils/AudioTransform.cpp
)

# 头文件
set(HEADERS
    src/audio/audio_input.h
    src/audio/audio_output.h
    src/audio/vad_detector.h
    src/asr/asr_engine.h
    # src/asr/feature_extractor.h  # Removed - whisper.cpp handles feature extraction
    src/query/query_engine.h
    src/query/sqlite_db.h
    src/query/llm_client.h
    src/tts/tts_engine.h
    src/tts/text_frontend.h
    src/utils/logger.h
    src/utils/config.h
    src/utils/thread_pool.h
    src/utils/AudioTransform.h
)

# 可执行文件
add_executable(voice_assistant ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(voice_assistant
    PRIVATE
        whisper
        onnxruntime  # For TTS (CosyVoice model inference)
        ${PORTAUDIO_LIBRARIES}
        SQLite::SQLite3
        Threads::Threads
        llama
        ggml
)

# yaml-cpp
if(YAML_CPP_FOUND)
    target_link_libraries(voice_assistant PRIVATE ${YAML_CPP_LIBRARIES})
    target_include_directories(voice_assistant PRIVATE ${YAML_CPP_INCLUDE_DIRS})
endif()

# spdlog
if(spdlog_FOUND)
    target_link_libraries(voice_assistant PRIVATE spdlog::spdlog)
endif()

# CUDA
if(USE_CUDA)
    target_link_libraries(voice_assistant PRIVATE CUDA::cudart)
endif()

# 编译选项
target_compile_options(voice_assistant PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    $<$<CONFIG:Release>:-march=native>
)

# 安装
install(TARGETS voice_assistant DESTINATION bin)
install(FILES config/config.yaml DESTINATION etc/voice_assistant)

# 测试
enable_testing()

# 单元测试 (可选)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# 打印配置信息
message(STATUS "========== Configuration ==========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Whisper.cpp: Enabled")
message(STATUS "PortAudio: ${PORTAUDIO_LIBRARIES}")
message(STATUS "SQLite3: ${SQLite3_VERSION}")
message(STATUS "CUDA support: ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
endif()
message(STATUS "===================================")
