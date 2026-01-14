#ifndef VOICE_ASSISTANT_AUDIO_INPUT_H
#define VOICE_ASSISTANT_AUDIO_INPUT_H

#include <portaudio.h>
#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <atomic>

namespace voice_assistant {
namespace audio {

/**
 * @brief 音频输入类，基于 PortAudio
 *
 * 负责从麦克风捕获音频数据并通过回调函数传递给上层
 */
class AudioInput {
public:
    using AudioCallback = std::function<void(const float* data, size_t frames)>;

    /**
     * @brief 构造函数
     * @param sample_rate 采样率 (默认 16000 Hz)
     * @param channels 声道数 (默认 1 单声道)
     * @param frames_per_buffer 每次回调的帧数 (默认 256)
     */
    AudioInput(int sample_rate = 16000,
               int channels = 1,
               unsigned long frames_per_buffer = 256);

    ~AudioInput();

    // 禁止拷贝
    AudioInput(const AudioInput&) = delete;
    AudioInput& operator=(const AudioInput&) = delete;

    /**
     * @brief 初始化音频输入
     * @param device_name 设备名称，空字符串使用默认设备
     * @return 成功返回 true
     */
    bool initialize(const std::string& device_name = std::string());

    /**
     * @brief 开始录音
     * @param callback 音频数据回调函数
     * @return 成功返回 true
     */
    bool start(AudioCallback callback);

    /**
     * @brief 停止录音
     */
    void stop();

    /**
     * @brief 检查是否正在录音
     */
    bool isActive() const { return is_active_; }

    /**
     * @brief 列出所有可用的输入设备
     */
    static std::vector<std::string> listDevices();

    /**
     * @brief 获取默认输入设备名称
     */
    static std::string getDefaultDevice();

private:
    /**
     * @brief PortAudio 回调函数
     */
    static int paCallback(const void* input_buffer,
                         void* output_buffer,
                         unsigned long frames_per_buffer,
                         const PaStreamCallbackTimeInfo* time_info,
                         PaStreamCallbackFlags status_flags,
                         void* user_data);

    /**
     * @brief 查找设备索引
     */
    int findDeviceIndex(const std::string& device_name);

    PaStream* stream_;
    int sample_rate_;
    int channels_;
    unsigned long frames_per_buffer_;
    std::atomic<bool> is_active_;
    AudioCallback callback_;
};

} // namespace audio
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_AUDIO_INPUT_H
