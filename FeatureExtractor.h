#pragma once

#include <cstddef>
#include <cmath>
#include <cstring>

struct AudioFeatures{
    float rms; //Signal Energy/Loudness
    float peak; //Peak , captures pick attacks
    float zcr; //How noisy how much it fluctuates (within one frame)
    float spectralCentroid; //Centre of mass of frequencies
    float spectralFlux; //Frequenecy changes between frames (Over frames)
}; 

class FeatureExtractor{
public:
    void Init(size_t frameSize, float samplingRate);

    void PushSample(float x);

    bool FrameReady() const;

    bool ProcessFrame(AudioFeatures& outFeatures);

    size_t GetFrameSize() const { return frameSize_; }

private:
    static constexpr size_t kMaxFrameSize = 512;
    static constexpr float  kPi = 3.14159265358979323846f;

    float captureBuffer_[kMaxFrameSize];
    float analysisBuffer_[kMaxFrameSize];

    float fftIn_[kMaxFrameSize];
    float prevMag_[kMaxFrameSize / 2];

    float window_[kMaxFrameSize];

    size_t frameSize_ = 256;
    size_t writePos_  = 0;

    float samplingRate_ = 48000.0f;

    bool frameReady_ = false;

    void ComputeHannWindow();
    void ComputeTimeDomainFeatures(const float* frame, AudioFeatures& features);
    void ComputeSpectralFeatures(const float* frame, AudioFeatures& features);
};