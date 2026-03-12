#include "daisy_pod.h"
#include "daisysp.h"
#include "FeatureExtractor.h"

using namespace daisy;
using namespace daisysp;
using namespace std;

volatile bool newFrameReady = false;
AudioFeatures latestFeatures;

DaisyPod hw;
FeatureExtractor extractor;
AudioFeatures features;

void AudioCallBack(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        float x = in[0][i];

        if(extractor.ProcessSample(x, features))
        {
            latestFeatures = features;
            newFrameReady = true;
        }

        out[0][i] = x;
        out[1][i] = x;
    }
}

int main(void)
{
    hw.Init();
    hw.seed.StartLog();
    System::Delay(3000);

    hw.SetAudioBlockSize(48);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    extractor.Init(256, 48000.0f);

    hw.seed.PrintLine("Daisy feature extractor starting...");
    hw.StartAdc();
    hw.StartAudio(AudioCallBack);
    hw.seed.PrintLine("Audio started");

    while(1)
    {
        if(newFrameReady)
        {
            newFrameReady = false;

            hw.seed.PrintLine("RMS: %d  Peak: %d  ZCR: %d  Centroid: %d  Flux: %d",
                  (int)(latestFeatures.rms * 1000),
                  (int)(latestFeatures.peak * 1000),
                  (int)(latestFeatures.zcr * 1000),
                  (int)(latestFeatures.spectralCentroid),
                  (int)(latestFeatures.spectralFlux * 1000));
        }

        System::Delay(50);
    }
}