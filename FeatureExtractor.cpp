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
    prevRms_      = 0.0f;
    prevEnvelope_ = 0.0f;
    prevPeak_     = 0.0f;
    historyIndex_ = 0;
    historyCount_ = 0;
    onsetIndex_ = 0;
    onsetCount_ = 0;
    elapsedTime_ = 0.0f;
    lastOnsetTime_ = -1000.0f;

    std::memset(captureBuffer_, 0, sizeof(captureBuffer_));
    std::memset(analysisBuffer_, 0, sizeof(analysisBuffer_));
    std::memset(fftIn_, 0, sizeof(fftIn_));
    std::memset(prevMag_, 0, sizeof(prevMag_));
    std::memset(window_, 0, sizeof(window_));
    std::memset(rmsHistory_, 0, sizeof(rmsHistory_));
    std::memset(centroidHistory_, 0, sizeof(centroidHistory_));
    std::memset(onsetTimes_, 0, sizeof(onsetTimes_));

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

    rmsHistory_[historyIndex_] = outFeatures.rms;
    centroidHistory_[historyIndex_] = outFeatures.spectralCentroid;

    historyIndex_ = (historyIndex_ + 1) % 8;
    if(historyCount_ < 8){
        historyCount_++;
    }

    float rmsMean = 0.0f;
    float centroidMean = 0.0f;

    for(int i = 0; i < historyCount_; i++){
        rmsMean += rmsHistory_[i];
        centroidMean += centroidHistory_[i];
    }

    if(historyCount_ > 0){
        rmsMean /= static_cast<float>(historyCount_);
        centroidMean /= static_cast<float>(historyCount_);
    }

    float rmsVar = 0.0f;
    float centroidVar = 0.0f;

    for(int i = 0; i < historyCount_; i++){
        float dr = rmsHistory_[i] - rmsMean;
        float dc = centroidHistory_[i] - centroidMean;
        rmsVar += dr * dr;
        centroidVar += dc * dc;
    }

    if(historyCount_ > 0){
        rmsVar /= static_cast<float>(historyCount_);
        centroidVar /= static_cast<float>(historyCount_);
    }

    outFeatures.rmsVariance = rmsVar;
    outFeatures.centroidVariance = centroidVar;

    float frameDuration = static_cast<float>(frameSize_) / samplingRate_;
    elapsedTime_ += frameDuration;

    float onsetThreshold = 0.020f;
    float peakRiseThreshold = 0.012f;
    float envelopeRiseThreshold = 0.008f;

    bool onsetDetected =
        outFeatures.peak > onsetThreshold &&
        (outFeatures.peak - prevPeak_) > peakRiseThreshold &&
        outFeatures.envelopeDelta > envelopeRiseThreshold;

    if(onsetDetected){
        onsetTimes_[onsetIndex_] = elapsedTime_;
        onsetIndex_ = (onsetIndex_ + 1) % 8;
        if(onsetCount_ < 8){
            onsetCount_++;
        }
        lastOnsetTime_ = elapsedTime_;
    }

    float onsetWindow = 1.0f;
    int recentOnsets = 0;
    for(int i = 0; i < onsetCount_; i++){
        if((elapsedTime_ - onsetTimes_[i]) <= onsetWindow){
            recentOnsets++;
        }
    }

    outFeatures.onsetCount = static_cast<float>(recentOnsets);

    if(lastOnsetTime_ > -999.0f){
        outFeatures.timeSinceLastOnset = elapsedTime_ - lastOnsetTime_;
    }
    else{
        outFeatures.timeSinceLastOnset = 1.0f;
    }

    if(outFeatures.timeSinceLastOnset < 0.0f){
        outFeatures.timeSinceLastOnset = 1.0f;
    }

    if(outFeatures.timeSinceLastOnset > 2.0f){
        outFeatures.timeSinceLastOnset = 2.0f;
    }

    prevPeak_ = outFeatures.peak;

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
    float peak = 0.0f;
    float crossings = 0.0f;
    float envelopeSum = 0.0f;

    for(size_t n = 0; n < frameSize_; n++)
    {
        float x = frame[n];
        sumSquares += x * x;

        float a = fabsf(x);
        envelopeSum += a;

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
    features.rmsDelta = fabsf(features.rms - prevRms_);
    features.envelope = envelopeSum / static_cast<float>(frameSize_);
    features.envelopeDelta = fabsf(features.envelope - prevEnvelope_);

    prevRms_ = features.rms;
    prevEnvelope_ = features.envelope;
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
        if(diff > 0.0f){
            spectralTotal += diff;
        }
        prevMag_[k] = magnitude;
    }

    if(runningDen > 1e-8f){
        features.spectralCentroid = runningNum / runningDen;
    }
    else{
        features.spectralCentroid = 0.0f;
    }

    features.spectralFlux = spectralTotal;

    if(features.spectralFlux > 5.0f){
        features.spectralFlux = 5.0f;
    }
}