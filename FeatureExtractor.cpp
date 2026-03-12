#include "FeatureExtractor.h"
#include <cstdio>

void FeatureExtractor::Init(size_t frameSize, float samplingRate)
{
    frameSize_ = frameSize;
    samplingRate_ = samplingRate;
    fftSize_ = frameSize;
    writePos_ = 0;

    std::memset(frame_, 0, sizeof(frame_));
    std::memset(prevMag_, 0, sizeof(prevMag_));
    std::memset(fftIn_, 0, sizeof(fftIn_));
    std::memset(fftOut_, 0, sizeof(fftOut_));

    ComputeHannWindow();

    arm_status result = arm_rfft_fast_init_f32(&fft_, fftSize_);
    if(result != ARM_MATH_SUCCESS)
    {
        printf("ARM FFT FAILED\n");
    }
}

bool FeatureExtractor::ProcessSample(float x, AudioFeatures& outFeatures)
{
    if(writePos_ < frameSize_)
    {
        frame_[writePos_++] = x;
    }

    if(writePos_ >= frameSize_)
    {
        ComputeFeatures(outFeatures);
        writePos_ = 0;
        return true;
    }

    return false;
}

void FeatureExtractor::ComputeHannWindow()
{
    for(size_t n = 0; n < frameSize_; n++)
    {
        window_[n] = 0.5f * (1.0f - std::cos((2.0f * kPi * static_cast<float>(n)))
                                      / static_cast<float>(frameSize_ - 1));
    }
}

void FeatureExtractor::ComputeFeatures(AudioFeatures& features)
{
    float total = 0.0f;
    for(size_t n = 0; n < frameSize_; n++)
    {
        float x = frame_[n];
        total += x * x;
    }
    features.rms = std::sqrt(total / static_cast<float>(frameSize_));

    float maximumAmplitude = 0.0f;
    for(size_t n = 0; n < frameSize_; n++)
    {
        float a = fabsf(frame_[n]);
        if(a > maximumAmplitude)
        {
            maximumAmplitude = a;
        }
    }
    features.peak = maximumAmplitude;

    float runningTotal = 0.0f;
    for(size_t n = 1; n < frameSize_; n++)
    {
        if(frame_[n] * frame_[n - 1] < 0.0f)
        {
            runningTotal += 1.0f;
        }
    }
    features.zcr = runningTotal / static_cast<float>(frameSize_ - 1);

    float runningNum = 0.0f;
    float runningDen = 0.0f;
    float spectralTotal = 0.0f;
    float binWidth = samplingRate_ / static_cast<float>(frameSize_);

    for(size_t n = 0; n < frameSize_; n++)
    {
        fftIn_[n] = frame_[n] * window_[n];
    }

    arm_rfft_fast_f32(&fft_, fftIn_, fftOut_, 0);

    for(size_t k = 1; k < (fftSize_ / 2); k++)
    {
        float real = fftOut_[2 * k];
        float imaginary = fftOut_[2 * k + 1];
        float magnitude = sqrtf(real * real + imaginary * imaginary);

        float fk = static_cast<float>(k) * binWidth;
        runningNum += fk * magnitude;
        runningDen += magnitude;

        float diff = magnitude - prevMag_[k];
        spectralTotal += diff * diff;
        prevMag_[k] = magnitude;
    }

    features.spectralCentroid = (runningDen > 0.0f) ? (runningNum / runningDen) : 0.0f;
    features.spectralFlux = spectralTotal;
}