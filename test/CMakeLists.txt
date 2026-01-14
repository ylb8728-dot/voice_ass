# 测试模块 CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

# 测试源文件公共依赖
set(TEST_COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/audio/audio_input.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/audio_output.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/vad_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/asr/asr_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/config.cpp
)

# 公共包含目录
set(TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${PORTAUDIO_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# ===== 测试1: AudioInput 测试 =====
add_executable(test_audio_input
    test_audio_input.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/audio_input.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_audio_input PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_audio_input
    PRIVATE
        ${PORTAUDIO_LIBRARIES}
        Threads::Threads
)

# ===== 测试2: VAD 测试 =====
add_executable(test_vad
    test_vad.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/vad_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_vad PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_vad
    PRIVATE
        Threads::Threads
)

# ===== 测试3: ASR Engine 测试 =====
add_executable(test_asr_engine
    test_asr_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/asr/asr_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_asr_engine PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_asr_engine
    PRIVATE
        whisper
        Threads::Threads
)

if(USE_CUDA)
    target_link_libraries(test_asr_engine PRIVATE CUDA::cudart)
endif()

# ===== 测试4: 综合测试 (Audio + VAD + ASR) =====
add_executable(test_audio_asr_pipeline
    test_audio_asr_pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/audio_input.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/vad_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/asr/asr_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/config.cpp
)

target_include_directories(test_audio_asr_pipeline PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_audio_asr_pipeline
    PRIVATE
        whisper
        ${PORTAUDIO_LIBRARIES}
        Threads::Threads
)

if(YAML_CPP_FOUND)
    target_link_libraries(test_audio_asr_pipeline PRIVATE ${YAML_CPP_LIBRARIES})
endif()

if(USE_CUDA)
    target_link_libraries(test_audio_asr_pipeline PRIVATE CUDA::cudart)
endif()

# ===== 测试5: SQLite 数据库测试 =====
add_executable(test_sqlite_db
    test_sqlite_db.cpp
    ${CMAKE_SOURCE_DIR}/src/query/sqlite_db.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_sqlite_db PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_sqlite_db
    PRIVATE
        SQLite::SQLite3
        Threads::Threads
)

# ===== 测试6: LLM 客户端测试 =====
add_executable(test_llm_client
    test_llm_client.cpp
    ${CMAKE_SOURCE_DIR}/src/query/llm_client.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_llm_client PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_llm_client
    PRIVATE
        llama
        ggml
        Threads::Threads
)

if(USE_CUDA)
    target_link_libraries(test_llm_client PRIVATE CUDA::cudart)
endif()

# ===== 测试7: QueryEngine 综合测试 =====
add_executable(test_query_engine
    test_query_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/query/query_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/query/sqlite_db.cpp
    ${CMAKE_SOURCE_DIR}/src/query/llm_client.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_query_engine PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_query_engine
    PRIVATE
        llama
        ggml
        SQLite::SQLite3
        Threads::Threads
)

if(USE_CUDA)
    target_link_libraries(test_query_engine PRIVATE CUDA::cudart)
endif()

# 编译选项
foreach(target test_audio_input test_vad test_asr_engine test_audio_asr_pipeline test_sqlite_db test_llm_client test_query_engine)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
endforeach()

# ===== 测试8: 完整流水线测试 (Audio + ASR + Query) =====
add_executable(test_full_pipeline
    test_full_pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/audio_input.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/vad_detector.cpp
    ${CMAKE_SOURCE_DIR}/src/asr/asr_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/query/query_engine.cpp
    ${CMAKE_SOURCE_DIR}/src/query/sqlite_db.cpp
    ${CMAKE_SOURCE_DIR}/src/query/llm_client.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/logger.cpp
)

target_include_directories(test_full_pipeline PRIVATE ${TEST_INCLUDE_DIRS})

target_link_libraries(test_full_pipeline
    PRIVATE
        whisper
        llama
        ggml
        ${PORTAUDIO_LIBRARIES}
        SQLite::SQLite3
        Threads::Threads
)

if(USE_CUDA)
    target_link_libraries(test_full_pipeline PRIVATE CUDA::cudart)
endif()

target_compile_options(test_full_pipeline PRIVATE -Wall -Wextra)

message(STATUS "Test executables configured:")
message(STATUS "  - test_audio_input")
message(STATUS "  - test_vad")
message(STATUS "  - test_asr_engine")
message(STATUS "  - test_audio_asr_pipeline")
message(STATUS "  - test_sqlite_db")
message(STATUS "  - test_llm_client")
message(STATUS "  - test_query_engine")
message(STATUS "  - test_full_pipeline")
