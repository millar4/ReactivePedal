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
    MODE_CHORUS,
    MODE_AMBIENT,
    MODE_REVERB
    
};

enum ClipMode {
    CLIP_BYPASS = 0,
    CLIP_SOFT,
    CLIP_HARD,
    CLIP_ASYM,
    CLIP_FUZZ
};

struct TonePreset{
    const char* toneName;
    float preamp;
    float drive;
    float tone;
    float mix;
    float level;
    float clipThreshold;
    ClipMode clipMode;
};


volatile WhichEffect currEffectMode = MODE_DISTORTION;
static constexpr int bufferSize = 48000;
DSY_SDRAM_BSS float delayBuffer[bufferSize];

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

volatile float smoothedDrive = 1.0f;
volatile float smoothedTone = 0.3f;
volatile float smoothedPreamp = 3.0f;
volatile float smoothedMix = 0.9f;
volatile float smoothedLevel = 0.8f;
volatile float smoothedClipThreshold = 1.0f;

bool useGeneratedSignal = true;
float signalPhase = 0.0f;
float signalTime = 0.0f;

float targetDrive = 1.0f;
float targetTone = 0.25f;
float targetPreamp = 3.0f;
float targetMix = 0.9f;
float targetLevel = 1.5f;
float targetClipThreshold = 1.0f;


float ambientMix = 0.92f;
float ambientFeedback = 0.88f;
float ambientRateHz = 0.03f;
float ambientDepthMs = 12.0f;
float ambientBaseDelayMs = 700.0f;
float ambientToneState = 0.0f;

float ambientPhase = 0.0f;
float ambientPhaseIncrement = 2.0f * 3.14159265359f * ambientRateHz / sampleRate;
float ambientBaseDelay = sampleRate * ambientBaseDelayMs / 1000.0f;
float ambientDepth = sampleRate * ambientDepthMs / 1000.0f;


float reverbMix = 0.55f;
float reverbFeedback = 0.72f;
float reverbToneState = 0.0f;

float reverbDelay1Ms = 140.0f;
float reverbDelay2Ms = 260.0f;
float reverbDelay3Ms = 420.0f;

float reverbDelay1 = sampleRate * reverbDelay1Ms / 1000.0f;
float reverbDelay2 = sampleRate * reverbDelay2Ms / 1000.0f;
float reverbDelay3 = sampleRate * reverbDelay3Ms / 1000.0f;

ClipMode currentClipMode = CLIP_SOFT;

float toneStateL = 0.0f;


static const TonePreset tonePresets[] = {
    {"Default", 3.0f, 2.0f, 0.25f, 0.92f, 1.5f, 1.0f, CLIP_SOFT},
};

static constexpr int noNumTonePresets = sizeof(tonePresets) / sizeof(tonePresets[0]);
int currentPresetIndex = 0;

float GenerateSignal();
float ambientProcessor(float x);
float reverbProcessor(float x);

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
    float pre = x * smoothedPreamp;
    float drive = pre * smoothedDrive;
    float wet = ApplyClipper(drive, currentClipMode, smoothedClipThreshold);
    float mixed = (1.0f - smoothedMix) * x + smoothedMix * wet;

    toneStateL = toneStateL + smoothedTone * (mixed - toneStateL);

    float y = toneStateL * smoothedLevel;
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
        float x = useGeneratedSignal ? GenerateSignal() : in[0][i];
        float y = x;

        extractor.PushSample(x);

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
        else if(currEffectMode == MODE_AMBIENT){
            y = ambientProcessor(x);
        }
        else if(currEffectMode == MODE_REVERB){
            y = reverbProcessor(x);
        }

        out[0][i] = y;
        out[1][i] = y;
    }
}

float ambientProcessor(float x){
    float delay = ambientBaseDelay + ambientDepth * sinf(ambientPhase);
    float delayed = ReadDelay(delay);

    ambientToneState = ambientToneState + 0.01f * (delayed - ambientToneState);

    float softened = 0.6f * delayed + 0.4f * ambientToneState;
    float fb = 0.2f * x + ambientFeedback * softened;

    if(fb > 1.0f){
        fb = 1.0f;
    }
    if(fb < -1.0f){
        fb = -1.0f;
    }

    delayBuffer[writeIndex] = fb;
    writeIndex = (writeIndex + 1) % bufferSize;

    ambientPhase += ambientPhaseIncrement;
    if(ambientPhase >= 6.28318530718f){
        ambientPhase -= 6.28318530718f;
    }

    float y = 0.15f * x + 0.85f * softened;
    return y;
}

float reverbProcessor(float x){
    float d1 = ReadDelay(reverbDelay1);
    float d2 = ReadDelay(reverbDelay2);
    float d3 = ReadDelay(reverbDelay3);

    float echoes = 0.55f * d1 + 0.30f * d2 + 0.15f * d3;

    reverbToneState = reverbToneState + 0.03f * (echoes - reverbToneState);

    float fb = 0.35f * x + reverbFeedback * reverbToneState;
    if(fb > 1.0f){
        fb = 1.0f;
    }
    if(fb < -1.0f){
        fb = -1.0f;
    }

    delayBuffer[writeIndex] = fb;
    writeIndex = (writeIndex + 1) % bufferSize;

    float y = (1.0f - reverbMix) * x + reverbMix * echoes;
    return y;
}

void AudioCallBack(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size){
        chorusSampler(in, out, size);
}

float GenerateSignal()
{
    float x;
    float envelope;
    float noise = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    if(currEffectMode == MODE_AMBIENT){
        envelope = expf(-0.12f * signalTime);

        float body = sinf(signalPhase);
        body += 0.01f * sinf(2.0f * signalPhase);
        body += 0.005f * sinf(3.0f * signalPhase);

        float pick = 0.08f * noise * expf(-12.0f * signalTime);

        x = (body * 0.35f + pick) * envelope;
    }
    else if(currEffectMode == MODE_REVERB){
        envelope = expf(-0.35f * signalTime);

        float body = sinf(signalPhase);
        body += 0.06f * sinf(2.0f * signalPhase);
        body += 0.025f * sinf(3.0f * signalPhase);
        body += 0.01f * sinf(4.0f * signalPhase);

        float pick = 0.12f * noise * expf(-15.0f * signalTime);

        x = (body * 0.45f + pick) * envelope;
    }
    else{
        envelope = expf(-2.0f * signalTime);

        float body = sinf(signalPhase);
        body += 0.22f * sinf(2.0f * signalPhase);
        body += 0.10f * sinf(3.0f * signalPhase);
        body += 0.04f * sinf(4.0f * signalPhase);

        float pick = 0.18f * noise * expf(-20.0f * signalTime);

        x = (body + pick) * envelope;
    }

    float drift = 1.0f + 0.002f * sinf(0.7f * signalTime);
    signalPhase += 2.0f * 3.14159265359f * 220.0f * drift / sampleRate;

    if(signalPhase >= 6.28318530718f){
        signalPhase -= 6.28318530718f;
    }

    signalTime += 1.0f / sampleRate;
    if(currEffectMode == MODE_AMBIENT){
        if(signalTime >= 5.0f){
            signalTime = 0.0f;
        }
    }
    else if(currEffectMode == MODE_REVERB){
        if(signalTime >= 3.0f){
            signalTime = 0.0f;
        }
    }
    else{
        if(signalTime >= 1.5f){
            signalTime = 0.0f;
        }
    }

    return x;
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
    int modeIndex = 0;

    while(1) {

        hw.ProcessAnalogControls();
        hw.ProcessDigitalControls();

        int inc = hw.encoder.Increment();
        if(inc > 0){
            modeIndex = (modeIndex + 1) % 4;
        }
        else if(inc < 0){
            modeIndex = (modeIndex + 3) % 4;
        }

        currEffectMode = static_cast<WhichEffect>(modeIndex);

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
            const char* modeName = "Distortion";
            if(currEffectMode == MODE_CHORUS){
                modeName = "Chorus";
            }
            else if(currEffectMode == MODE_AMBIENT){
                modeName = "Ambient";
            }
            else if(currEffectMode == MODE_REVERB){
                modeName = "Reverb";
            }

            hw.seed.PrintLine("Mode:%s Preset:%s RMS:%d Peak:%d ZCR:%d Cent:%dHz Flux:%d Drive:%d Tone:%d Pre:%d Mix:%d Level:%d",
                  modeName,
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