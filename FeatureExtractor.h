#pragma once

#include <cstddef>
#include <cmath>
#include <cstring>

struct AudioFeatures{
    float rms;
    float peak;
    float zcr;
    float spectralCentroid;
    float spectralFlux;
    float rmsDelta;
    float envelope;
    float envelopeDelta;
    float rmsVariance;
    float centroidVariance;
    float onsetCount;
    float timeSinceLastOnset;
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
    float prevRms_ = 0.0f;
    float prevEnvelope_ = 0.0f;
    float prevPeak_ = 0.0f;

    float rmsHistory_[8];
    float centroidHistory_[8];
    int historyIndex_ = 0;
    int historyCount_ = 0;

    float onsetTimes_[8];
    int onsetIndex_ = 0;
    int onsetCount_ = 0;
    float elapsedTime_ = 0.0f;
    float lastOnsetTime_ = -1000.0f;

    bool frameReady_ = false;

    void ComputeHannWindow();
    void ComputeTimeDomainFeatures(const float* frame, AudioFeatures& features);
    void ComputeSpectralFeatures(const float* frame, AudioFeatures& features);
};