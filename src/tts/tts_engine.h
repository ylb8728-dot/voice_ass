#ifndef VOICE_ASSISTANT_TTS_ENGINE_H
#define VOICE_ASSISTANT_TTS_ENGINE_H

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "utils/AudioTransform.h"  // For iSTFT

namespace voice_assistant {
namespace tts {

// 前向声明
class TextFrontend;

/**
 * @brief TTS 引擎类，基于 ONNX Runtime
 *
 * 封装 CosyVoice2 模型，提供离线和流式语音合成功能
 */
class TTSEngine {
public:
    using StreamCallback = std::function<void(const float* audio, size_t num_samples)>;

    struct Config {
        std::string model_dir;
        std::string speaker = "中文女";
        bool enable_streaming = true;
        int streaming_chunk_size = 512;
        int num_threads = 4;
        bool use_gpu = true;
        int gpu_device_id = 0;
        float speed = 1.0f;
        float pitch = 1.0f;
        float energy = 1.0f;
    };

    explicit TTSEngine(const Config& config);
    ~TTSEngine();

    // 禁止拷贝
    TTSEngine(const TTSEngine&) = delete;
    TTSEngine& operator=(const TTSEngine&) = delete;

    /**
     * @brief 初始化 TTS 引擎
     * @return 成功返回 true
     */
    bool initialize();

    /**
     * @brief 离线合成（完整文本）
     * @param text 待合成文本
     * @return 音频数据 (22.05kHz, float32)
     */
    std::vector<float> synthesize(const std::string& text);

    /**
     * @brief 流式合成（低延迟）
     * @param text 待合成文本
     * @param callback 音频流回调函数
     */
    void synthesizeStream(const std::string& text, StreamCallback callback);

    /**
     * @brief 设置说话人
     */
    void setSpeaker(const std::string& speaker) {
        config_.speaker = speaker;
    }

    /**
     * @brief 设置语速
     */
    void setSpeed(float speed) {
        config_.speed = speed;
    }

    /**
     * @brief 检查是否初始化成功
     */
    bool isInitialized() const { return initialized_; }

private:
    /**
     * @brief 文本预处理
     */
    std::string preprocessText(const std::string& text);

    /**
     * @brief 文本转 token
     */
    std::vector<int64_t> textToTokens(const std::string& text);

    /**
     * @brief 运行 LLM 模型
     */
    std::vector<float> runLLM(const std::vector<int64_t>& tokens);

    /**
     * @brief 运行 Flow 模型
     */
    std::vector<float> runFlow(const std::vector<float>& llm_output);

    /**
     * @brief 运行 Vocoder 模型
     */
    std::vector<float> runVocoder(const std::vector<float>& mel_spec);

    Config config_;
    bool initialized_;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> llm_session_;
    std::unique_ptr<Ort::Session> flow_session_;
    std::unique_ptr<Ort::Session> vocoder_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::MemoryInfo memory_info_;

    // 文本前端
    std::unique_ptr<TextFrontend> text_frontend_;

    // 音频变换（用于 iSTFT）
    std::unique_ptr<utils::AudioTransform> audio_transform_;
};

} // namespace tts
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_TTS_ENGINE_H
