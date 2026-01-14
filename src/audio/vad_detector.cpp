#include "vad_detector.h"
#include "../utils/logger.h"
#include <cmath>
#include <algorithm>

namespace voice_assistant {
namespace audio {

VADDetector::VADDetector()
    : config_(Config())
    , is_in_speech_(false)
    , speech_frame_count_(0)
    , silence_frame_count_(0) {
}

VADDetector::VADDetector(const Config& config)
    : config_(config)
    , is_in_speech_(false)
    , speech_frame_count_(0)
    , silence_frame_count_(0) {
}

VADDetector::~VADDetector() = default;

bool VADDetector::isSpeech(const float* audio, size_t num_samples) {
    if (!audio || num_samples == 0) {
        return false;
    }

    // 计算能量和过零率
    float energy = calculateEnergy(audio, num_samples);
    float zcr = calculateZeroCrossingRate(audio, num_samples);

    // 简单的能量和过零率判断
    bool has_speech = (energy > config_.energy_threshold) &&
                      (zcr < config_.zero_crossing_threshold);

    if (has_speech) {
        speech_frame_count_++;
        silence_frame_count_ = 0;

        // 连续语音帧超过阈值，进入语音段
        if (speech_frame_count_ >= config_.min_speech_frames) {
            if (!is_in_speech_) {
                is_in_speech_ = true;
                LOG_DEBUG("Speech started");
            }
        }
    } else {
        silence_frame_count_++;
        speech_frame_count_ = 0;

        // 连续静音帧超过阈值，退出语音段
        if (silence_frame_count_ >= config_.min_silence_frames) {
            if (is_in_speech_) {
                is_in_speech_ = false;
                LOG_DEBUG("Speech ended");
            }
        }
    }

    return is_in_speech_;
}

void VADDetector::reset() {
    is_in_speech_ = false;
    speech_frame_count_ = 0;
    silence_frame_count_ = 0;
}

float VADDetector::calculateEnergy(const float* audio, size_t num_samples) {
    float sum = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        sum += audio[i] * audio[i];
    }
    return sum / num_samples;
}

float VADDetector::calculateZeroCrossingRate(const float* audio, size_t num_samples) {
    if (num_samples < 2) {
        return 0.0f;
    }

    size_t zero_crossings = 0;
    for (size_t i = 1; i < num_samples; ++i) {
        if ((audio[i] >= 0 && audio[i - 1] < 0) ||
            (audio[i] < 0 && audio[i - 1] >= 0)) {
            zero_crossings++;
        }
    }

    return static_cast<float>(zero_crossings) / (num_samples - 1);
}

} // namespace audio
} // namespace voice_assistant
