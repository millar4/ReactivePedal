#include "daisy_pod.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include "daisysp.h"
#include "arm_math.h"

using namespace daisy;
using namespace std;
using namespace daisysp;

//create a struct for my results
struct AudioFeatures
{
    float rms; //average signal energy 
    float peak; //largest sample mag
    float zcr; //zero crossing rate 
    float spectralCentroid; //Spectral centre of mass
    float spectralFlux; //How much spectrum has changed since previous frame
};

class FeatureExtractor{
    public:
    void Init(size_t frameSize, float samplingRate);
    bool ProcessSample(float x, AudioFeatures& outFeatures);


    private:
    static constexpr size_t kMaxFrameSize = 512;

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

void FeatureExtractor::Init(size_t frameSize, float samplingRate){
    frameSize_ = frameSize;
    samplingRate_ = samplingRate;
    fftSize_ = frameSize;
    writePos_ = 0;

    std::memset(frame_, 0, sizeof(frame_));
    std::memset(prevMag_, 0, sizeof(prevMag_));
    std::memset(fftIn_, 0, sizeof(fftIn_));
    std::memset(fftOut_, 0, sizeof(fftOut_));

    ComputeHannWindow();

    arm_status result = arm_rfft_fast_init_f32(&fft_, frameSize_);
        if(result != ARM_MATH_SUCCESS)
    {
        printf("ARM FFT FAILED\n");
    }
}

bool FeatureExtractor::ProcessSample(float x, AudioFeatures& outFeatures){

    if(writePos_ < frameSize_){
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

//use hann window to create smoothed edges to reduce the effect of spectral leakage
void FeatureExtractor::ComputeHannWindow(){
    for(size_t  n = 0; n < frameSize_; n++){
        window_[n] = 0.5f * (1.0f - std::cos((2.0f * M_PI * n) / (frameSize_ - 1))); //creates our windowed signal: x[n]w[n]
    }
}

void FeatureExtractor::ComputeFeatures(AudioFeatures& features){
   //compute our RMS
   float total = 0;
   for(size_t n = 0; n < frameSize_; n++){
        float x = frame_[n];
        total += x * x;
   }
   features.rms = sqrt(total /frameSize_);

   //Peak amplutude
   float maximumAmplitude = 0.0f;
   for(size_t n = 0; n < frameSize_; n++){
    float a = fabsf(frame_[n]);
    if(a > maximumAmplitude){
        maximumAmplitude = a;
    }
   }
   features.peak = maximumAmplitude;
   
   //Zero crossing rate
    float runningTotal = 0.0f;
    for(size_t n = 1; n < frameSize_; n++)
    {
        if(frame_[n] * frame_[n-1] < 0)
            runningTotal += 1.0f;
    }
    features.zcr = runningTotal / (frameSize_ - 1);

    //Spectral centroid + flux
    float runningNum = 0.0f;
    float runningDen = 0.0f;
    float spectralTotal = 0.0f;
    float binWidth = samplingRate_ / frameSize_;

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

        float fk = k * binWidth;
        runningNum += fk * magnitude;
        runningDen += magnitude;

        float diff = magnitude - prevMag_[k];
        spectralTotal += diff * diff;
        prevMag_[k] = magnitude;
    }

    features.spectralCentroid = (runningDen > 0.0f) ? (runningNum / runningDen) : 0.0f;
    features.spectralFlux = spectralTotal;
}