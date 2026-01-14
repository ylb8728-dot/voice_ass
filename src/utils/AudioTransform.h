// AudioTransform.h - STFT/iSTFT utilities for audio processing
#pragma once

#include <vector>
#include <complex>
#include <cmath>

namespace voice_assistant {
namespace utils {

/**
 * @brief Audio transformation utilities (STFT/iSTFT)
 *
 * Implements Short-Time Fourier Transform and its inverse for audio processing.
 * Used primarily for vocoder (HiFT) post-processing.
 */
class AudioTransform {
public:
    /**
     * @brief Construct AudioTransform with specified parameters
     * @param n_fft FFT size (default: 16 for HiFT)
     * @param hop_length Hop size between frames (default: 4 for HiFT)
     * @param window_type Window type: "hann", "hamming", or "rectangular" (default: "hann")
     */
    explicit AudioTransform(int n_fft = 16, int hop_length = 4,
                          const std::string& window_type = "hann");

    /**
     * @brief Perform inverse STFT to reconstruct audio from magnitude and phase
     *
     * @param magnitude Magnitude spectrum [n_fft/2+1, num_frames]
     * @param phase Phase spectrum [n_fft/2+1, num_frames]
     * @param num_frames Number of time frames
     * @return Reconstructed audio signal
     */
    std::vector<float> istft(const std::vector<float>& magnitude,
                            const std::vector<float>& phase,
                            int num_frames);

    /**
     * @brief Perform STFT on audio signal
     *
     * @param audio Input audio signal
     * @param magnitude [out] Magnitude spectrum
     * @param phase [out] Phase spectrum
     * @return Number of frames
     */
    int stft(const std::vector<float>& audio,
            std::vector<float>& magnitude,
            std::vector<float>& phase);

    // Getters
    int getNFFT() const { return n_fft_; }
    int getHopLength() const { return hop_length_; }

private:
    /**
     * @brief Generate window function
     */
    void generateWindow();

    /**
     * @brief Perform DFT on a single frame
     * @param frame Time-domain frame (windowed)
     * @param spectrum [out] Complex spectrum
     */
    void dft(const std::vector<float>& frame,
            std::vector<std::complex<float>>& spectrum);

    /**
     * @brief Perform inverse DFT on a single frame
     * @param spectrum Complex spectrum (positive frequencies only)
     * @param frame [out] Time-domain frame
     */
    void idft(const std::vector<std::complex<float>>& spectrum,
             std::vector<float>& frame);

    int n_fft_;           // FFT size
    int hop_length_;      // Hop size
    std::string window_type_; // Window type
    std::vector<float> window_;  // Window function

    // Pre-computed DFT matrices for efficiency
    std::vector<std::complex<float>> dft_matrix_;      // [n_fft, n_fft]
    std::vector<std::complex<float>> idft_matrix_;     // [n_fft, n_fft/2+1]
};

} // namespace utils
} // namespace voice_assistant
