// AudioTransform.cpp - STFT/iSTFT implementation
#include "AudioTransform.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace voice_assistant {
namespace utils {

constexpr float PI = 3.14159265358979323846f;

AudioTransform::AudioTransform(int n_fft, int hop_length, const std::string& window_type)
    : n_fft_(n_fft), hop_length_(hop_length), window_type_(window_type) {

    if (n_fft <= 0 || hop_length <= 0) {
        throw std::invalid_argument("n_fft and hop_length must be positive");
    }

    generateWindow();

    // Pre-compute DFT matrix for efficiency
    // DFT matrix: W[k,n] = exp(-2j*pi*k*n/N)
    dft_matrix_.resize(n_fft_ * n_fft_);
    for (int k = 0; k < n_fft_; ++k) {
        for (int n = 0; n < n_fft_; ++n) {
            float angle = -2.0f * PI * k * n / n_fft_;
            dft_matrix_[k * n_fft_ + n] = std::complex<float>(
                std::cos(angle), std::sin(angle)
            );
        }
    }

    // Pre-compute IDFT matrix (only positive frequencies)
    // IDFT matrix: W[n,k] = exp(2j*pi*k*n/N) / N
    int num_bins = n_fft_ / 2 + 1;
    idft_matrix_.resize(n_fft_ * num_bins);
    for (int n = 0; n < n_fft_; ++n) {
        for (int k = 0; k < num_bins; ++k) {
            float angle = 2.0f * PI * k * n / n_fft_;
            idft_matrix_[n * num_bins + k] = std::complex<float>(
                std::cos(angle), std::sin(angle)
            ) / static_cast<float>(n_fft_);
        }
    }
}

void AudioTransform::generateWindow() {
    window_.resize(n_fft_);

    if (window_type_ == "hann") {
        // Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
        for (int n = 0; n < n_fft_; ++n) {
            window_[n] = 0.5f * (1.0f - std::cos(2.0f * PI * n / (n_fft_ - 1)));
        }
    } else if (window_type_ == "hamming") {
        // Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n/(N-1))
        for (int n = 0; n < n_fft_; ++n) {
            window_[n] = 0.54f - 0.46f * std::cos(2.0f * PI * n / (n_fft_ - 1));
        }
    } else if (window_type_ == "rectangular") {
        // Rectangular window
        std::fill(window_.begin(), window_.end(), 1.0f);
    } else {
        throw std::invalid_argument("Unknown window type: " + window_type_);
    }
}

void AudioTransform::dft(const std::vector<float>& frame,
                        std::vector<std::complex<float>>& spectrum) {
    if (static_cast<int>(frame.size()) != n_fft_) {
        throw std::invalid_argument("Frame size must equal n_fft");
    }

    int num_bins = n_fft_ / 2 + 1;
    spectrum.resize(num_bins);

    // Compute DFT using pre-computed matrix (only positive frequencies)
    for (int k = 0; k < num_bins; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int n = 0; n < n_fft_; ++n) {
            sum += frame[n] * dft_matrix_[k * n_fft_ + n];
        }
        spectrum[k] = sum;
    }
}

void AudioTransform::idft(const std::vector<std::complex<float>>& spectrum,
                         std::vector<float>& frame) {
    int num_bins = n_fft_ / 2 + 1;
    if (static_cast<int>(spectrum.size()) != num_bins) {
        throw std::invalid_argument("Spectrum size must equal n_fft/2+1");
    }

    frame.resize(n_fft_);

    // Compute IDFT using pre-computed matrix
    for (int n = 0; n < n_fft_; ++n) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int k = 0; k < num_bins; ++k) {
            sum += spectrum[k] * idft_matrix_[n * num_bins + k];
        }
        // Take real part only (imaginary should be ~0 for real signals)
        frame[n] = sum.real();

        // Account for symmetry: add contribution from negative frequencies
        // For k > 0 and k < N/2, the negative frequency contributes conjugate
        for (int k = 1; k < num_bins - 1; ++k) {
            float angle = -2.0f * PI * k * n / n_fft_;
            std::complex<float> neg_freq = std::conj(spectrum[k]) * std::complex<float>(
                std::cos(angle), std::sin(angle)
            ) / static_cast<float>(n_fft_);
            frame[n] += neg_freq.real();
        }
    }
}

std::vector<float> AudioTransform::istft(const std::vector<float>& magnitude,
                                        const std::vector<float>& phase,
                                        int num_frames) {
    int num_bins = n_fft_ / 2 + 1;

    // Validate input dimensions
    if (static_cast<int>(magnitude.size()) != num_bins * num_frames ||
        static_cast<int>(phase.size()) != num_bins * num_frames) {
        throw std::invalid_argument("Magnitude and phase dimensions mismatch");
    }

    // Calculate output signal length
    int signal_length = (num_frames - 1) * hop_length_ + n_fft_;
    std::vector<float> signal(signal_length, 0.0f);
    std::vector<float> window_sum(signal_length, 0.0f);  // For overlap-add normalization

    // Process each frame
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        // Reconstruct complex spectrum from magnitude and phase
        std::vector<std::complex<float>> spectrum(num_bins);
        for (int k = 0; k < num_bins; ++k) {
            int idx = k * num_frames + frame_idx;
            float mag = std::min(magnitude[idx], 100.0f);  // Clamp magnitude
            float ph = phase[idx];

            // Complex number from polar form: mag * exp(j*phase)
            spectrum[k] = std::complex<float>(
                mag * std::cos(ph),
                mag * std::sin(ph)
            );
        }

        // Perform IDFT to get time-domain frame
        std::vector<float> frame;
        idft(spectrum, frame);

        // Apply window and overlap-add
        int start_idx = frame_idx * hop_length_;
        for (int n = 0; n < n_fft_; ++n) {
            if (start_idx + n < signal_length) {
                signal[start_idx + n] += frame[n] * window_[n];
                window_sum[start_idx + n] += window_[n] * window_[n];
            }
        }
    }

    // Normalize by window sum to compensate for overlap
    for (int i = 0; i < signal_length; ++i) {
        if (window_sum[i] > 1e-8f) {
            signal[i] /= window_sum[i];
        }
    }

    // Apply final clipping
    for (float& sample : signal) {
        sample = std::max(-0.99f, std::min(0.99f, sample));
    }

    return signal;
}

int AudioTransform::stft(const std::vector<float>& audio,
                        std::vector<float>& magnitude,
                        std::vector<float>& phase) {
    int signal_length = audio.size();
    int num_frames = (signal_length - n_fft_) / hop_length_ + 1;

    if (num_frames <= 0) {
        throw std::invalid_argument("Audio signal too short for STFT");
    }

    int num_bins = n_fft_ / 2 + 1;
    magnitude.resize(num_bins * num_frames);
    phase.resize(num_bins * num_frames);

    // Process each frame
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        // Extract frame
        std::vector<float> frame(n_fft_);
        int start_idx = frame_idx * hop_length_;
        for (int n = 0; n < n_fft_; ++n) {
            frame[n] = audio[start_idx + n] * window_[n];
        }

        // Perform DFT
        std::vector<std::complex<float>> spectrum;
        dft(frame, spectrum);

        // Extract magnitude and phase
        for (int k = 0; k < num_bins; ++k) {
            int idx = k * num_frames + frame_idx;
            magnitude[idx] = std::abs(spectrum[k]);
            phase[idx] = std::arg(spectrum[k]);
        }
    }

    return num_frames;
}

} // namespace utils
} // namespace voice_assistant
