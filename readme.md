  #编译测试

  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
  make -j$(nproc)

  #运行测试

  1. AudioInput 音频输入测试
  ./test/test_audio_input --list      # 列出音频设备
  ./test/test_audio_input --record 5  # 录音5秒
  ./test/test_audio_input --monitor   # 实时音量监测

  2. VAD 语音活动检测测试
  ./test/test_vad --all  # 运行所有VAD测试

  3. ASR 引擎测试
  ./build/test/test_asr_engine --model whisper.cpp/models/ggml-large-v3-turbo.bin --all
  ./build/test/test_asr_engine --model whisper.cpp/models/ggml-large-v3-turbo.bin --wav test.wav

  4. 综合流水线测试（实时语音识别）
  ./build/test/test_audio_asr_pipeline --model whisper.cpp/models/ggml-large-v3-turbo.bin

  5. 简单录音识别测试
  ./build/test/test_audio_asr_pipeline --model whisper.cpp/models/ggml-large-v3-turbo.bin --simple 10

  6. SQLite 数据库测试 (不需要模型)
  cd /home/ye/project/vass/voice_ass
  ./build/test/test_sqlite_db --all          # 运行所有测试
  ./build/test/test_sqlite_db --search       # 全文检索测试
  ./build/test/test_sqlite_db --perf         # 性能测试

  7. LLM 客户端测试
  ./build/test/test_llm_client --model llama.cpp/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf --init   # 初始化测试
  ./build/test/test_llm_client --model llama.cpp/models/Qwen3-8B-Q8_0.gguf --basic  # 基本查询
  ./build/test/test_llm_client --model llama.cpp/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf --bench  # 性能基准
  ./build/test/test_llm_client --remote      # 远程LLM测试(占位实现)

  8. QueryEngine 综合测试
  ./build/test/test_query_engine --local     # LOCAL_ONLY模式(不需要LLM模型)
  ./build/test/test_query_engine --model models/qwen2-7b-instruct-q4_0.gguf --llm      # LLM_ONLY模式
  ./build/test/test_query_engine --model models/qwen2-7b-instruct-q4_0.gguf --hybrid   # 混合模式
  ./build/test/test_query_engine --model llama.cpp/models/Qwen3-8B-Q8_0.gguf --interactive  # 交互式测试

  注意: 请根据实际LLM模型路径调整 --model 参数。

  9. 完整流水线测试方法 (语音+ASR+Query)

  9.1. LOCAL模式 (仅本地知识库，响应快，不需要LLM模型)
  ./build/test/test_full_pipeline \
      --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
      --mode local

  9.2. LLM模式 (使用LLM回答所有问题)
  ./build/test/test_full_pipeline \
      --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
      --llm-model models/qwen2-7b-instruct-q4_0.gguf \
      --mode llm

  9.3. HYBRID模式 (优先本地知识库，不确定时用LLM)
  ./build/test/test_full_pipeline \
      --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
      --llm-model llama.cpp/models/Qwen3-8B-Q8_0.gguf \
      --mode hybrid


