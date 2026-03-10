#include "daisy_pod.h"
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace daisy;


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

    float frame_[kMaxFrameSize];
    float prevMag_[kMaxFrameSize / 2];
    float window_[kMaxFrameSize];

    size_t frameSize_ = 0;
    size_t writePos_ = 0;
    float samplingRate_ = 48000.0f;

    void ComputeHannWindow();
    void ComputeFeatures(AudioFeatures& features);
};

void FeatureExtractor::Init(size_t frameSize, float samplingRate){
    frameSize_ = frameSize;
    samplingRate_ = samplingRate;
    writePos_ = 0;

    std::memset(frame_, 0, sizeof(frame_));
    std::memset(prevMag_, 0, sizeof(prevMag_));

    ComputeHannWindow();
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
}