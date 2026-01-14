#ifndef VOICE_ASSISTANT_VAD_DETECTOR_H
#define VOICE_ASSISTANT_VAD_DETECTOR_H

#include <vector>
#include <cstddef>

namespace voice_assistant {
namespace audio {

/**
 * @brief VAD (Voice Activity Detection) 检测器
 *
 * 使用简单的能量阈值方法检测语音活动
 */
class VADDetector {
public:
    struct Config {
        float energy_threshold = 0.01f;      // 能量阈值
        float zero_crossing_threshold = 0.3f; // 过零率阈值
        size_t min_speech_frames = 10;        // 最小语音帧数
        size_t min_silence_frames = 20;       // 最小静音帧数
        size_t frame_size = 256;              // 帧大小（样本数）
    };

    VADDetector();
    explicit VADDetector(const Config& config);
    ~VADDetector();

    /**
     * @brief 检测音频帧是否包含语音
     * @param audio 音频数据 (float32)
     * @param num_samples 样本数
     * @return true 如果检测到语音
     */
    bool isSpeech(const float* audio, size_t num_samples);

    /**
     * @brief 检测音频帧是否包含语音（vector 接口）
     */
    bool isSpeech(const std::vector<float>& audio) {
        return isSpeech(audio.data(), audio.size());
    }

    /**
     * @brief 重置检测器状态
     */
    void reset();

    /**
     * @brief 获取当前状态（是否在语音段）
     */
    bool isInSpeech() const { return is_in_speech_; }

private:
    /**
     * @brief 计算音频能量
     */
    float calculateEnergy(const float* audio, size_t num_samples);

    /**
     * @brief 计算过零率
     */
    float calculateZeroCrossingRate(const float* audio, size_t num_samples);

    Config config_;
    bool is_in_speech_;
    size_t speech_frame_count_;
    size_t silence_frame_count_;
};

} // namespace audio
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_VAD_DETECTOR_H
