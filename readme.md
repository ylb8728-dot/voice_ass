
#硬件要求

    CPU: 支持AVX2指令集（推荐）

    内存: 至少8GB RAM（运行大模型需要16GB+）

    存储: 至少10GB可用空间用于模型文件

    音频: 支持麦克风输入

#软件依赖

确保以下依赖已正确安装：

    CMake 3.10+

    GCC 8+ 或 Clang 10+

    SQLite3 3.25+

    PortAudio（音频输入）

    Whisper.cpp（语音识别）

    llama.cpp（大语言模型）

#模型文件

下载以下模型文件到指定目录：
bash

##ASR模型（选择其一）
wget -P whisper.cpp/models/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
wget -P whisper.cpp/models/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

##LLM模型（选择其一）
wget -P llama.cpp/models/ https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q8_0.gguf
wget -P llama.cpp/models/ https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/qwen2.5-vl-7b-instruct-q8_0.gguf

#编译与构建
##步骤1：创建构建目录
bash

mkdir -p build
cd build

##步骤2：配置CMake
bash

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DWITH_LLM_SUPPORT=ON \
  -DWITH_ASR_SUPPORT=ON

##步骤3：编译项目
bash

###使用所有CPU核心编译
make -j$(nproc)

###或指定核心数
make -j4

步骤4：验证编译结果
bash

###检查生成的可执行文件
ls -la test/test_*

#模块测试
##音频输入测试
测试项	命令	说明
列出设备	./test/test_audio_input --list	显示所有可用音频设备
录音测试	./test/test_audio_input --record 5	录音5秒并保存为test.wav
音量监测	./test/test_audio_input --monitor	实时显示输入音量
设备测试	./test/test_audio_input --test-device 0	测试指定设备（0为默认）
##语音活动检测测试
bash

###运行所有VAD测试
./test/test_vad --all

###测试特定音频文件
./test/test_vad --file test.wav

###性能测试
./test/test_vad --benchmark --duration 30

##ASR引擎测试
bash

###基本配置
ASR_MODEL="whisper.cpp/models/ggml-large-v3-turbo.bin"

###运行所有测试
./build/test/test_asr_engine --model $ASR_MODEL --all

###测试特定音频文件
./build/test/test_asr_engine --model $ASR_MODEL --wav samples/english.wav

###测试不同语言
./build/test/test_asr_engine --model $ASR_MODEL --lang zh --wav samples/chinese.wav

###性能基准测试
./build/test/test_asr_engine --model $ASR_MODEL --benchmark

##实时语音识别流水线
bash

###实时识别测试
./build/test/test_audio_asr_pipeline \
  --model $ASR_MODEL \
  --vad-threshold 0.5

###简单录音识别（录音10秒）
./build/test/test_audio_asr_pipeline \
  --model $ASR_MODEL \
  --simple 10

###输出到文件
./build/test/test_audio_asr_pipeline \
  --model $ASR_MODEL \
  --output transcript.txt \
  --duration 30

##SQLite数据库测试
bash

###切换到项目目录
cd /home/ye/project/vass/voice_ass

###运行所有测试
./build/test/test_sqlite_db --all

###数据库初始化测试
./build/test/test_sqlite_db --init

###全文检索测试
./build/test/test_sqlite_db --search --query "关键词"

###性能测试（1000条记录）
./build/test/test_sqlite_db --perf --records 1000

###清理测试数据库
./build/test/test_sqlite_db --clean

##LLM客户端测试
bash

###设置模型路径
LLM_MODEL="llama.cpp/models/Qwen2.5-7B-Instruct-Q8_0.gguf"
VL_MODEL="llama.cpp/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf"

###初始化测试
./build/test/test_llm_client --model $LLM_MODEL --init

###基本查询测试
./build/test/test_llm_client --model $LLM_MODEL --basic

###多轮对话测试
./build/test/test_llm_client --model $LLM_MODEL --chat

###视觉语言模型测试（需要图像输入）
./build/test/test_llm_client --model $VL_MODEL --image test.jpg --prompt "描述这张图片"

###性能基准测试
./build/test/test_llm_client --model $LLM_MODEL --bench --tokens 100

###远程API测试
./build/test/test_llm_client --remote \
  --api-key "your-api-key" \
  --endpoint "https://api.openai.com/v1"

##QueryEngine查询引擎测试
bash

###LOCAL模式（仅本地知识库）
./build/test/test_query_engine --local

###LLM模式（使用LLM回答）
./build/test/test_query_engine --model $LLM_MODEL --llm

###混合模式
./build/test/test_query_engine --model $LLM_MODEL --hybrid

###交互式测试
./build/test/test_query_engine --model $LLM_MODEL --interactive

###特定查询测试
./build/test/test_query_engine --model $LLM_MODEL --query "什么是人工智能？"

###带上下文的查询
./build/test/test_query_engine --model $LLM_MODEL \
  --query "继续上面的讨论" \
  --context "之前我们讨论了机器学习的基本概念"

##完整流水线测试
配置参数说明
参数	说明	示例值
--asr-model	ASR模型路径	whisper.cpp/models/ggml-large-v3-turbo.bin
--llm-model	LLM模型路径	llama.cpp/models/Qwen3-8B-Q8_0.gguf
--mode	运行模式	local/llm/hybrid
--language	识别语言	zh/en/auto
--device	音频设备ID	0（默认设备）
--duration	最长运行时间	300（5分钟）
--log-level	日志级别	info/debug/error
测试命令
###模式1：LOCAL模式

仅使用本地知识库，响应最快
bash

./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --mode local \
  --language zh \
  --log-level info \
  --duration 180

###模式2：LLM模式

使用LLM回答所有问题
bash

./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --llm-model llama.cpp/models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  --mode llm \
  --language auto \
  --log-level debug \
  --device 0

###模式3：HYBRID模式

混合模式，优先本地知识库
bash

./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --llm-model llama.cpp/models/Qwen3-8B-Q8_0.gguf \
  --mode hybrid \
  --confidence-threshold 0.7 \
  --max-local-results 3

###模式4：批处理模式

处理预录音频文件
bash

./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --llm-model llama.cpp/models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  --batch-file audio_list.txt \
  --output results.json \
  --workers 2