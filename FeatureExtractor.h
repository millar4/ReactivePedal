#pragma once

#include <cstddef>
#include <cmath>
#include <cstring>
#include "arm_math.h"

struct AudioFeatures
{
    float rms;
    float peak;
    float zcr;
    float spectralCentroid;
    float spectralFlux;
};

class FeatureExtractor
{
public:
    void Init(size_t frameSize, float samplingRate);
    bool ProcessSample(float x, AudioFeatures& outFeatures);

private:
    static constexpr size_t kMaxFrameSize = 512;
    static constexpr float kPi = 3.14159265358979323846f;

    arm_rfft_fast_instance_f32 fft_;
    float fftIn_[kMaxFrameSize];
    float fftOut_[kMaxFrameSize];
    float frame_[kMaxFrameSize];
    float prevMag_[kMaxFrameSize / 2];
    float window_[kMaxFrameSize];

    size_t frameSize_ = 0;
    size_t writePos_ = 0;
    size_t fftSize_ = 0;
    float samplingRate_ = 48000.0f;

    void ComputeHannWindow();
    void ComputeFeatures(AudioFeatures& features);
};