#ifndef VOICE_ASSISTANT_ASR_ENGINE_H
#define VOICE_ASSISTANT_ASR_ENGINE_H

extern "C" {
    #include "whisper.h"
}

#include <string>
#include <vector>
#include <memory>

namespace voice_assistant {
namespace asr {

/**
 * @brief ASR 引擎类，基于 whisper.cpp
 *
 * 封装 Whisper large-v3 模型，提供离线和流式识别功能
 */
class ASREngine {
public:
    struct Config {
        std::string model_path;
        std::string language = "zh";   // ISO 639-1 语言代码
        int num_threads = 4;
        bool use_gpu = true;
        int gpu_device_id = 0;

        // Whisper 特有参数
        bool translate = false;         // 是否翻译为英文
        bool no_timestamps = true;      // 禁用时间戳（更快）
        float temperature = 0.0f;       // 解码温度（0=贪婪）
    };

    explicit ASREngine(const Config& config);
    ~ASREngine();

    // 禁止拷贝
    ASREngine(const ASREngine&) = delete;
    ASREngine& operator=(const ASREngine&) = delete;

    /**
     * @brief 初始化 ASR 引擎
     * @return 成功返回 true
     */
    bool initialize();

    /**
     * @brief 离线识别（完整音频）
     * @param audio 音频数据 (16kHz, float32)
     * @param num_samples 样本数
     * @return 识别文本
     */
    std::string recognize(const float* audio, size_t num_samples);

    /**
     * @brief 离线识别（vector 接口）
     */
    std::string recognize(const std::vector<float>& audio) {
        return recognize(audio.data(), audio.size());
    }

    /**
     * @brief 开始流式识别
     */
    void startStream();

    /**
     * @brief 输入音频流
     * @param audio 音频数据
     * @param num_samples 样本数
     */
    void feedAudio(const float* audio, size_t num_samples);

    /**
     * @brief 获取部分识别结果
     * @return 当前识别的文本
     */
    std::string getPartialResult();

    /**
     * @brief 结束流式识别并获取最终结果
     * @return 最终识别文本
     */
    std::string endStream();

    /**
     * @brief 检查是否初始化成功
     */
    bool isInitialized() const { return initialized_; }

private:
    Config config_;
    bool initialized_;

    // Whisper.cpp 上下文和状态
    whisper_context* ctx_;
    whisper_state* state_;

    // 流式识别缓冲区
    std::vector<float> stream_buffer_;
    std::string partial_result_;
};

} // namespace asr
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_ASR_ENGINE_H
