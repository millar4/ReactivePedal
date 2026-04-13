#include "daisy_pod.h"
#include "daisysp.h"
#include "FeatureExtractor.h"
#include <cmath>

using namespace daisy;
using namespace daisysp;
using namespace std;

DaisyPod hw;
FeatureExtractor extractor;
AudioFeatures latestFeatures;
AudioFeatures smoothedFeatures = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};


enum WhichEffect{
    MODE_DISTORTION = 0,
    MODE_CHORUS
};

enum ClipMode {
    CLIP_BYPASS = 0, //We dont clip
    CLIP_SOFT, //smoothoverdrive 
    CLIP_HARD, //sharp clipping
    CLIP_ASYM, //positive and negative sides clip differently 
    CLIP_FUZZ //non linear fuzz shampe
};

struct TonePreset{
    const char* toneName; //name for our tone
    float preamp; //gain before distorition
    float drive; //extra gain into clip stage
    float tone; //low pass filter strength
    float mix; //dry and wet tone blend
    float level; //final output volume
    float clipThreshold; //clip ceiling
    ClipMode clipMode; //which clip function to use
};


volatile WhichEffect currEffectMode = MODE_DISTORTION;
static constexpr int bufferSize = 2048;
float delayBuffer[bufferSize];
int writeIndex = 0;

float sampleRate = 48000.0f;
float baseDelayMs = 12.0f;
float depthMs = 4.0f;
float rateHz = 0.6f;
float chorusMix = 0.4f;

float baseDelay = sampleRate * baseDelayMs / 1000.0f;
float depth = sampleRate * depthMs / 1000.0f;

float phase = 0.0f;
float phaseIncrement = 2.0f * 3.14159265359f * rateHz / sampleRate;
//smoothed to avoid strange behaviours with jumps

volatile float smoothedDrive = 1.0f;
volatile float smoothedTone = 0.3f;
volatile float smoothedPreamp = 3.0f;
volatile float smoothedMix = 0.9f;
volatile float smoothedLevel = 0.8f;
volatile float smoothedClipThreshold = 1.0f;

//move toward these
float targetDrive = 1.0f;
float targetTone = 0.25f;
float targetPreamp = 3.0f;
float targetMix = 0.9f;
float targetLevel = 1.5f;
float targetClipThreshold = 1.0f;

//Current clip mode, default to soft
ClipMode currentClipMode = CLIP_SOFT;

//Internal memory of the tone filter
float toneStateL = 0.0f;


//array where we store values for our presets
static const TonePreset tonePresets[] = {
    {"Default", 3.0f, 2.0f, 0.25f, 0.92f, 1.5f, 1.0f, CLIP_SOFT},
};

//how many pre sets are we storing?
static constexpr int noNumTonePresets = sizeof(tonePresets) / sizeof(tonePresets[0]);
int currentPresetIndex = 0;


//use later for hard clipping 
float Clamp(float x, float lo, float hi){
    if(x < lo){
        return lo;
    }
    if(x > hi){
        return hi;
    }
    return x;
}


void SmoothFeatures(const AudioFeatures& in, AudioFeatures& out){
    out.rms              = 0.8f  * out.rms + 0.2f  * in.rms;
    out.peak             = 0.8f  * out.peak + 0.2f  * in.peak;
    out.zcr              = 0.85f * out.zcr + 0.15f * in.zcr;
    out.spectralCentroid = 0.9f  * out.spectralCentroid + 0.1f * in.spectralCentroid;
    out.spectralFlux     = 0.95f * out.spectralFlux + 0.05f * in.spectralFlux;
}

float MySoftClip(float x){
    return x / (1.0f + fabsf(x));
}

float HardClip(float x, float threshold){
    return Clamp(x, -threshold, threshold);
}

float AsymClip(float x){
    if(x >= 0.0f){
        return tanhf(2.5f * x);
    }
    return tanhf(1.5f * x);
}

float FuzzClip(float x){

    float x3 = x * x * x;
    return tanhf(3.0f * x3);
}

float ApplyClipper(float x, ClipMode mode, float threshold){

    switch(mode){
        case CLIP_BYPASS:
            return x;
        case CLIP_SOFT:
            return MySoftClip(x);
        case CLIP_HARD:
            return HardClip(x, threshold);
        case CLIP_ASYM:
            return AsymClip(x);
        case CLIP_FUZZ:
            return FuzzClip(x);
        default:
            return x;
    }
}

void SetTonePreset(const TonePreset& preset){
    targetPreamp = preset.preamp;
    targetDrive = preset.drive;
    targetTone = preset.tone;
    targetMix = preset.mix;
    targetLevel = preset.level;
    targetClipThreshold = preset.clipThreshold;
    currentClipMode = preset.clipMode;
}

float ProcessTone(float x)
{
    float pre = x * smoothedPreamp; //pre amp stage, boost of reduce raw guitar signal 
    float drive = pre * smoothedDrive; //drive stage
    float wet = ApplyClipper(drive, currentClipMode, smoothedClipThreshold); //change waveform
    float mixed = (1.0f - smoothedMix) * x + smoothedMix * wet; //blend clean signal back in for clarity

    toneStateL = toneStateL + smoothedTone * (mixed - toneStateL); //lpf , further signal smoothing, remove high freq when smoothedTone is smaller

    float y = toneStateL * smoothedLevel; //final volume contorl
    return y;
}


float ReadDelay(float delay)
{
    float readPos = (float)writeIndex - delay;

    while(readPos < 0.0f)
    {
        readPos += bufferSize;
    }

    while(readPos >= bufferSize)
    {
        readPos -= bufferSize;
    }

    int index0 = (int)readPos;
    int index1 = (index0 + 1) % bufferSize;

    float frac = readPos - (float)index0;

    float s0 = delayBuffer[index0];
    float s1 = delayBuffer[index1];

    return s0 + frac * (s1 - s0);
}


void chorusSampler(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out , size_t size ){
    
    for(size_t i = 0; i < size; i++){
        float x = in[0][i];
        float y = x;

        if(currEffectMode == MODE_DISTORTION){
            y = ProcessTone(x);
        }
        else if(currEffectMode == MODE_CHORUS){
            float delay = baseDelay + depth * sinf(phase);
            float delayed = ReadDelay(delay);

            y = ((1.0f - chorusMix) * x + chorusMix * delayed) * 1.5f;

            delayBuffer[writeIndex] = x;
            writeIndex = (writeIndex + 1) % bufferSize;

            phase += phaseIncrement;
            if(phase >= 6.28318530718f){
                phase -= 6.28318530718f;
            }
        }

        out[0][i] = y;
        out[1][i] = y;
    }
}

void AudioCallBack(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size){
        chorusSampler(in, out, size);
}

int main(void){
    hw.Init();
    hw.seed.StartLog();
    System::Delay(3000);

    hw.SetAudioBlockSize(96);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    extractor.Init(256, 48000.0f);

    SetTonePreset(tonePresets[currentPresetIndex]);

    hw.seed.PrintLine("Starting...");
    hw.seed.PrintLine("Preset: %s", tonePresets[currentPresetIndex].toneName);
    hw.StartAudio(AudioCallBack);
    hw.seed.PrintLine("Audio started");

    uint32_t lastPrintTime = 0;

    while(1) {

        hw.ProcessAnalogControls();
        hw.ProcessDigitalControls();

        float mode = hw.knob1.Value();
        if(mode < 0.5f){
        currEffectMode = MODE_DISTORTION;
            }
        else{
                currEffectMode = MODE_CHORUS;
            }

        if(extractor.ProcessFrame(latestFeatures)) {
            SmoothFeatures(latestFeatures, smoothedFeatures);

            smoothedDrive = 0.90f * smoothedDrive + 0.10f * targetDrive;
            smoothedTone = 0.92f * smoothedTone + 0.08f * targetTone;
            smoothedPreamp = 0.90f * smoothedPreamp + 0.10f * targetPreamp;
            smoothedMix = 0.90f * smoothedMix + 0.10f * targetMix;
            smoothedLevel = 0.90f * smoothedLevel + 0.10f * targetLevel;
            smoothedClipThreshold = 0.90f * smoothedClipThreshold + 0.10f * targetClipThreshold;
        }

        uint32_t now = System::GetNow();
        if(now - lastPrintTime >= 1000) {
            lastPrintTime = now;

            hw.seed.PrintLine("Preset:%s RMS:%d Peak:%d ZCR:%d Cent:%dHz Flux:%d Drive:%d Tone:%d Pre:%d Mix:%d Level:%d",
                              tonePresets[currentPresetIndex].toneName,
                              (int)(smoothedFeatures.rms * 1000.0f),
                              (int)(smoothedFeatures.peak * 1000.0f),
                              (int)(smoothedFeatures.zcr * 1000.0f),
                              (int)(smoothedFeatures.spectralCentroid),
                              (int)(smoothedFeatures.spectralFlux * 1000.0f),
                              (int)(smoothedDrive * 100.0f),
                              (int)(smoothedTone * 1000.0f),
                              (int)(smoothedPreamp * 100.0f),
                              (int)(smoothedMix * 100.0f),
                              (int)(smoothedLevel * 100.0f));
        }
    }
}
