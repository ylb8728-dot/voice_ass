# CosyVoice2 C++实现与测试文档

本文档说明如何使用C++实现的CosyVoice2 TTS引擎进行Zero-Shot语音合成。

## 目录结构

```
src/tts/
├── cosyvoice2_engine.h         # CosyVoice2引擎头文件
├── cosyvoice2_engine.cpp       # CosyVoice2引擎实现
├── test_cosyvoice2.cpp         # 独立测试程序
├── CMakeLists.txt              # 测试程序构建配置
└── README_COSYVOICE2.md        # 本文档

scripts/
└── export_cosyvoice2_to_onnx.py  # PyTorch模型导出ONNX脚本
```

## 任务一：C++实现与测试

### 1.1 架构设计

CosyVoice2引擎实现了完整的Zero-Shot TTS pipeline：

```
输入文本 + Prompt音频
    ↓
Frontend (特征提取)
    ├── Text Tokenizer (Qwen tokenizer)
    ├── CamPLUS (Speaker embedding)
    ├── Speech Tokenizer (Audio → Speech tokens)
    └── Mel Spectrogram (Audio → Mel features)
    ↓
LLM (生成Speech tokens)
    ↓
Flow (Speech tokens → Mel spectrogram)
    ├── Flow Encoder
    └── Flow Decoder (CFM)
    ↓
HiFT Vocoder (Mel → Waveform)
    ↓
输出音频 (24kHz)
```

### 1.2 依赖项

- **ONNX Runtime** (>= 1.15.0)：用于模型推理
- **C++17编译器**：GCC 7+ 或 Clang 5+
- **CMake** (>= 3.18)
- **CUDA** (可选)：用于GPU加速

安装ONNX Runtime：
```bash
# 下载并解压到 /usr/local/onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-gpu-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.16.0.tgz
sudo mv onnxruntime-linux-x64-gpu-1.16.0 /usr/local/onnxruntime
```

### 1.3 模型准备

在运行C++程序前，需要准备ONNX模型文件。模型目录结构应为：

```
models/cosyvoice2/
├── campplus.onnx                # Speaker embedding提取
├── speech_tokenizer_v2.onnx     # 语音token提取
├── llm.onnx                     # LLM模型
├── flow_encoder.onnx            # Flow编码器
├── flow_decoder.onnx            # Flow解码器
├── hift.onnx                    # Vocoder
└── tokenizer.json               # 文本tokenizer
```

**注意**：`campplus.onnx`和`speech_tokenizer_v2.onnx`可以直接从CosyVoice2预训练模型目录复制。其他模型需要使用任务二的导出脚本生成。

### 1.4 编译测试程序

```bash
cd src/tts
mkdir build && cd build

# Release构建（使用GPU）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 生成可执行文件：test_cosyvoice2
```

### 1.5 运行测试

#### 准备Prompt音频

Prompt音频要求：
- 格式：WAV (16-bit PCM)
- 采样率：16000 Hz
- 声道：单声道
- 时长：3-10秒（不超过30秒）

可以使用ffmpeg转换：
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 zero_shot_prompt.wav
```

#### 运行示例

```bash
./test_cosyvoice2 \
    --model-dir /path/to/models/cosyvoice2 \
    --prompt-wav zero_shot_prompt.wav \
    --prompt-text "你好，这是一段测试语音，用于提取说话人的声音特征" \
    --tts-text "欢迎使用CosyVoice2语音合成系统，这是一个强大的零样本语音克隆工具" \
    --output output.wav \
    --stream \
    --gpu
```

参数说明：
- `--model-dir`：ONNX模型目录
- `--prompt-wav`：Prompt音频文件（16kHz WAV）
- `--prompt-text`：Prompt音频对应的文本
- `--tts-text`：要合成的目标文本
- `--output`：输出音频文件（24kHz WAV）
- `--stream`：启用流式合成（可选）
- `--gpu`：使用GPU加速（可选，默认启用）
- `--cpu`：仅使用CPU

#### 性能配置

在`cosyvoice2_engine.h`的Config结构体中，可以调整：

```cpp
Config config;
config.model_dir = "models/cosyvoice2";
config.use_gpu = true;              // 使用GPU
config.gpu_device_id = 0;           // GPU设备ID
config.num_threads = 4;             // CPU线程数
config.enable_streaming = true;     // 流式模式
config.chunk_size = 25;             // Stream chunk大小
config.speed = 1.0f;                // 语速（1.0=正常）
```

### 1.6 集成到主项目

要在主语音助手项目中使用CosyVoice2引擎：

1. 在`CMakeLists.txt`中添加源文件：
```cmake
set(SOURCES
    # ... 其他文件 ...
    src/tts/cosyvoice2_engine.cpp
)
```

2. 在代码中使用：
```cpp
#include "tts/cosyvoice2_engine.h"

// 初始化
tts::CosyVoice2Engine::Config config;
config.model_dir = "models/cosyvoice2";
config.use_gpu = true;

tts::CosyVoice2Engine engine(config);
engine.initialize();

// 合成
std::vector<float> audio = engine.synthesizeZeroShot(
    "要合成的文本",
    "提示文本",
    prompt_audio_data,
    prompt_audio_length
);
```

### 1.7 已知限制

当前实现的以下部分为占位实现，需要进一步完善：

1. **Tokenizer**：使用占位实现，需要集成真正的Qwen tokenizer
   - 建议使用 [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)

2. **Mel Spectrogram提取**：需要实现完整的STFT和Mel滤波器组
   - 参数：n_fft=1920, hop_size=480, win_size=1920, num_mels=80

3. **Speech Tokenizer前处理**：需要实现Whisper的log_mel_spectrogram

4. **LLM推理**：需要实现自回归token生成循环

5. **Streaming推理**：需要实现真正的chunk-by-chunk流式生成

## 任务二：PyTorch到ONNX导出

### 2.1 导出脚本说明

`scripts/export_cosyvoice2_to_onnx.py`脚本可将CosyVoice2的PyTorch模型导出为ONNX格式。

### 2.2 使用方法

#### 环境准备

```bash
cd CosyVoice
pip install -r requirements.txt
pip install onnx onnxruntime-gpu
```

#### 运行导出

```bash
python ../scripts/export_cosyvoice2_to_onnx.py \
    --model-dir pretrained_models/CosyVoice2-0.5B \
    --output-dir ../models/cosyvoice2_onnx \
    --opset-version 18 \
    --skip-llm  # LLM导出较复杂，暂时跳过
```

#### 导出的模型

- **flow_encoder.onnx**：Flow的Conformer编码器
- **flow_decoder.onnx**：Flow的CFM解码器（estimator）
- **hift.onnx**：HiFT vocoder

**注意**：LLM模型的导出需要特殊处理，因为它包含自回归生成逻辑。建议使用以下方案之一：

1. 使用TensorRT-LLM（推荐）
2. 导出为静态ONNX（固定sequence length）
3. 使用vLLM backend

### 2.3 模型验证

导出后会自动验证：
- ONNX模型结构完整性
- 输入输出形状和类型
- PyTorch vs ONNX输出一致性（数值误差<1e-3）

### 2.4 复制预构建模型

```bash
# 复制已有的ONNX模型
cp CosyVoice/pretrained_models/CosyVoice2-0.5B/campplus.onnx \
   models/cosyvoice2_onnx/

cp CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx \
   models/cosyvoice2_onnx/
```

## 配置文件参考

基于`CosyVoice/pretrained_models/CosyVoice2-0.5B/cosyvoice2.yaml`的关键配置：

```yaml
sample_rate: 24000
chunk_size: 25              # Streaming chunk大小（token）
token_frame_rate: 25        # Token帧率
token_mel_ratio: 2          # Token到Mel的比例

llm:
  speech_token_size: 6561   # Speech token词表大小
  llm_input_size: 896
  llm_output_size: 896

flow:
  input_size: 512
  output_size: 80
  spk_embed_dim: 192        # Speaker embedding维度

hift:
  in_channels: 80
  sampling_rate: 24000
  upsample_rates: [8, 5, 3] # 总上采样倍率 = 120 (对应hop_size=200)
```

## 开发计划

- [ ] 完善Tokenizer集成（使用tokenizers-cpp）
- [ ] 实现Mel spectrogram提取
- [ ] 实现LLM自回归生成
- [ ] 实现真正的streaming推理
- [ ] 添加语速、音调控制
- [ ] 支持batch推理
- [ ] 性能优化（TensorRT加速）
- [ ] 添加单元测试

## 参考资料

- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [CosyVoice2 Paper](https://arxiv.org/abs/2409.10759)

## 故障排除

### Q: ONNX Runtime找不到CUDA

A: 确保安装的是GPU版本的ONNX Runtime，并且CUDA版本匹配。可以检查：
```bash
ldd /usr/local/onnxruntime/lib/libonnxruntime.so | grep cuda
```

### Q: 模型加载失败

A: 检查模型文件路径是否正确，文件是否完整。可以使用onnx验证：
```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

### Q: 生成的音频质量差

A: 这可能是因为占位实现导致的。需要等待完整实现或使用Python版本进行对比。

## 联系方式

如有问题或建议，请联系项目维护者。
