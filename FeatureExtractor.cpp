#include "FeatureExtractor.h"

void FeatureExtractor::Init(size_t frameSize, float samplingRate){
    if(frameSize > kMaxFrameSize) {
        frameSize = kMaxFrameSize;
    }

    if(frameSize != 256) {

        frameSize = 256;
    }

    frameSize_    = frameSize;
    samplingRate_ = samplingRate;
    writePos_     = 0;
    frameReady_   = false;

    std::memset(captureBuffer_, 0, sizeof(captureBuffer_));
    std::memset(analysisBuffer_, 0, sizeof(analysisBuffer_));
    std::memset(fftIn_, 0, sizeof(fftIn_));
    std::memset(prevMag_, 0, sizeof(prevMag_));
    std::memset(window_, 0, sizeof(window_));

    ComputeHannWindow();
}


void FeatureExtractor::PushSample(float x){
    if(frameReady_){
        return;
    }

    captureBuffer_[writePos_++] = x;

    if(writePos_ >= frameSize_){
        std::memcpy(analysisBuffer_, captureBuffer_, frameSize_ * sizeof(float));
        writePos_   = 0;
        frameReady_ = true;
    }
}

bool FeatureExtractor::FrameReady() const{
    return frameReady_;
}

bool FeatureExtractor::ProcessFrame(AudioFeatures& outFeatures){
    if(!frameReady_){
        return false;
    }

    ComputeTimeDomainFeatures(analysisBuffer_, outFeatures);
    ComputeSpectralFeatures(analysisBuffer_, outFeatures);

    frameReady_ = false;
    return true;
}

void FeatureExtractor::ComputeHannWindow(){
    for(size_t n = 0; n < frameSize_; n++)
    {
        window_[n] = 0.5f * (1.0f - cosf((2.0f * kPi * static_cast<float>(n))
                                         / static_cast<float>(frameSize_ - 1)));
    }
}

void FeatureExtractor::ComputeTimeDomainFeatures(const float* frame, AudioFeatures& features){
    float sumSquares = 0.0f;
    float peak       = 0.0f;
    float crossings  = 0.0f;

    for(size_t n = 0; n < frameSize_; n++)
    {
        float x = frame[n];
        sumSquares += x * x;

        float a = fabsf(x);
        if(a > peak){
            peak = a;
        }

        if(n > 0){
            if(frame[n] * frame[n - 1] < 0.0f){
                crossings += 1.0f;
            }
        }
    }
    features.rms  = sqrtf(sumSquares / static_cast<float>(frameSize_));
    features.peak = peak;
    features.zcr  = crossings / static_cast<float>(frameSize_ - 1);
}

void FeatureExtractor::ComputeSpectralFeatures(const float* frame, AudioFeatures& features){
    for(size_t n = 0; n < frameSize_; n++){
        fftIn_[n] = frame[n] * window_[n];
    }

    float runningNum = 0.0f;
    float runningDen = 0.0f;
    float spectralTotal = 0.0f;
    float binWidth = samplingRate_ / static_cast<float>(frameSize_);

    for(size_t k = 1; k < (frameSize_ / 2); k++)
    {
        float real = 0.0f;
        float imag = 0.0f;

        for(size_t n = 0; n < frameSize_; n++){
            float angle = 2.0f * kPi * static_cast<float>(k * n)
                        / static_cast<float>(frameSize_);
            real += fftIn_[n] * cosf(angle);
            imag -= fftIn_[n] * sinf(angle);
        }

        float magnitude = sqrtf(real * real + imag * imag);
        float freq = static_cast<float>(k) * binWidth;

        runningNum += freq * magnitude;
        runningDen += magnitude;

        float diff = magnitude - prevMag_[k];
        spectralTotal += diff * diff;
        prevMag_[k] = magnitude;
    }

    if(runningDen > 1e-8f){
        features.spectralCentroid = runningNum / runningDen;
    }
    else{
        features.spectralCentroid = 0.0f;
    }
    features.spectralFlux = spectralTotal;
}