#include "daisy_pod.h"
#include "daisysp.h"
#include "FeatureExtractor.h"
#include "NeuralNet.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "MiniCABClassifier.h"
#include "screen.h"

using namespace daisy;
using namespace daisysp;
using namespace std;

DaisyPod hw;
FeatureExtractor extractor;
AudioFeatures latestFeatures;
AudioFeatures smoothedFeatures = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

NeuralNet neuralnet;
NNOutput nnOutput = {};
MappingState savedMapping;
MiniCABClassifier minicab;

bool hasSavedNeuralNetMapping = false;
bool hasSavedCABMapping = false;
bool predictionMode = false;
bool isTraining = false;

int currentLabel = 0;
int predictedCandidate = 0;
int predictedStableClass = 0;
int predictedHoldCount = 0;
int stableClassAge = 0;
int candidateClassAge = 0;

static constexpr int perClassBufferCapacity = 32;
static constexpr int cabWindowSize = 32;
static constexpr int cabExamplesPerClass = 32;
static constexpr int minCABExamplesPerClassForTraining = 12;

DSY_SDRAM_BSS AudioFeatures featureBuffer[4][perClassBufferCapacity];
DSY_SDRAM_BSS AudioFeatures cabWindow[cabWindowSize];
DSY_SDRAM_BSS AudioFeatures cabTrainBuffer[4][cabExamplesPerClass][cabWindowSize];

int bufferIndex[4] = {0, 0, 0, 0};
int bufferCount[4] = {0, 0, 0, 0};

int cabWindowIndex = 0;
int cabWindowCount = 0;
int cabTrainIndex[4] = {0, 0, 0, 0};
int cabTrainCount[4] = {0, 0, 0, 0};

uint32_t lastCABStoreMs = 0;
uint32_t lastCABPredictMs = 0;

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

enum NetworkMode{
    NETWORK_NEURALNET = 0,
    NETWORK_MINICAB
};

NetworkMode currentNetworkMode = NETWORK_NEURALNET;

CABOutput cabOutput = {};

volatile WhichEffect currEffectMode = MODE_DISTORTION;
WhichEffect lastAppliedEffectMode = MODE_DISTORTION;

static constexpr int bufferSize = 48000;
DSY_SDRAM_BSS float delayBuffer[bufferSize];

int writeIndex = 0;

float sampleRate = 48000.0f;
float baseDelayMs = 18.0f;
float depthMs = 1.2f;
float rateHz = 0.28f;
float chorusMix = 0.32f;

float ambientLevel = 4.0f;
float reverbLevel = 3.5f;

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

bool useGeneratedSignal = false;
float signalPhase = 0.0f;
float signalTime = 0.0f;

float targetDrive = 1.0f;
float targetTone = 0.25f;
float targetPreamp = 3.0f;
float targetMix = 0.9f;
float targetLevel = 1.5f;
float targetClipThreshold = 1.0f;

float ambientMix = 0.97f;
float ambientFeedback = 0.93f;
float ambientRateHz = 0.03f;
float ambientDepthMs = 12.0f;
float ambientBaseDelayMs = 700.0f;
float ambientToneState = 0.0f;

float ambientPhase = 0.0f;
float ambientPhaseIncrement = 2.0f * 3.14159265359f * ambientRateHz / sampleRate;
float ambientBaseDelay = sampleRate * ambientBaseDelayMs / 1000.0f;
float ambientDepth = sampleRate * ambientDepthMs / 1000.0f;

float reverbMix = 0.82f;
float reverbFeedback = 0.84f;
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
    {"Distortion", 4.4f, 4.6f, 0.14f, 0.94f, 1.18f, 0.78f, CLIP_ASYM},
    {"Chorus", 1.2f, 1.0f, 0.32f, 0.45f, 1.2f, 1.0f, CLIP_BYPASS},
    {"Ambient", 1.1f, 0.9f, 0.20f, 0.55f, 1.0f, 1.0f, CLIP_SOFT},
    {"Reverb", 1.4f, 1.1f, 0.24f, 0.65f, 1.1f, 1.0f, CLIP_SOFT},
};

static constexpr int noNumTonePresets = sizeof(tonePresets) / sizeof(tonePresets[0]);
int currentPresetIndex = 0;

float GenerateSignal();
float ambientProcessor(float x);
float reverbProcessor(float x);

bool phraseActive = false;
uint32_t phraseStartMs = 0;
uint32_t lastActiveMs = 0;
AudioFeatures phraseAccum = {};
int phraseFrames = 0;
static constexpr uint32_t phraseEndGapMs = 100;
static constexpr uint32_t maxPhraseLengthMs = 650;
static constexpr int minPhraseFrames = 3;
static constexpr uint32_t phraseAttackCooldownMs = 180;

AudioFeatures lastPhraseFeatures = {};
bool lastPhraseReady = false;

float phraseMaxRms = 0.0f;
float phraseMinRms = 1.0f;
float phraseMaxCentroid = 0.0f;
float phraseMinCentroid = 1000000.0f;
float phraseMaxPeak = 0.0f;
int phraseAttackCount = 0;
uint32_t lastPhraseAttackMs = 0;
uint32_t firstPhraseAttackMs = 0;
uint32_t prevPhraseAttackMs = 0;
float phraseAttackGapSumMs = 0.0f;
int phraseAttackGapCount = 0;

static constexpr int targetPhrasesPerClass = 32;
uint32_t lastAcceptedPhraseMs[4] = {0, 0, 0, 0};

float classDistances[4] = {9999.0f, 9999.0f, 9999.0f, 9999.0f};
float classScores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float smoothedClassScores[4] = {0.25f, 0.25f, 0.25f, 0.25f};

float datasetMean[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float datasetStd[12] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

bool CurrentNetworkHasSavedMapping(){
    return currentNetworkMode == NETWORK_MINICAB ? hasSavedCABMapping : hasSavedNeuralNetMapping;
}

void ForceScreenRedraw(int& lastMenuIndex,
                       int& lastPreset,
                       int& lastTraining,
                       int& lastPredictionMode,
                       int& lastNetworkMode,
                       int& lastSaved,
                       int& lastB0,
                       int& lastB1,
                       int& lastB2,
                       int& lastB3){
    lastMenuIndex = -1;
    lastPreset = -1;
    lastTraining = -1;
    lastPredictionMode = -1;
    lastNetworkMode = -1;
    lastSaved = -1;
    lastB0 = -1;
    lastB1 = -1;
    lastB2 = -1;
    lastB3 = -1;
}

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
    out.rms                = 0.35f * out.rms + 0.65f * in.rms;
    out.peak               = 0.35f * out.peak + 0.65f * in.peak;
    out.zcr                = 0.45f * out.zcr + 0.55f * in.zcr;
    out.spectralCentroid   = 0.45f * out.spectralCentroid + 0.55f * in.spectralCentroid;
    out.spectralFlux       = 0.30f * out.spectralFlux + 0.70f * in.spectralFlux;
    out.rmsDelta           = 0.25f * out.rmsDelta + 0.75f * in.rmsDelta;
    out.envelope           = 0.35f * out.envelope + 0.65f * in.envelope;
    out.envelopeDelta      = 0.25f * out.envelopeDelta + 0.75f * in.envelopeDelta;
    out.rmsVariance        = 0.45f * out.rmsVariance + 0.55f * in.rmsVariance;
    out.centroidVariance   = 0.45f * out.centroidVariance + 0.55f * in.centroidVariance;
    out.onsetCount         = 0.15f * out.onsetCount + 0.85f * in.onsetCount;
    out.timeSinceLastOnset = 0.15f * out.timeSinceLastOnset + 0.85f * in.timeSinceLastOnset;
}

void ResetCABWindow(){
    cabWindowIndex = 0;
    cabWindowCount = 0;
    std::memset(cabWindow, 0, sizeof(cabWindow));
}

void PushCABWindowFrame(const AudioFeatures& f){
    cabWindow[cabWindowIndex] = f;
    cabWindowIndex = (cabWindowIndex + 1) % cabWindowSize;

    if(cabWindowCount < cabWindowSize){
        cabWindowCount++;
    }
}

bool CABWindowHasSignal(){
    if(cabWindowCount < cabWindowSize){
        return false;
    }

    int activeFrames = 0;
    float maxPeak = 0.0f;
    float rmsSum = 0.0f;
    float fluxSum = 0.0f;
    float envSum = 0.0f;

    for(int i = 0; i < cabWindowSize; i++){
        rmsSum += cabWindow[i].rms;
        fluxSum += cabWindow[i].spectralFlux;
        envSum += cabWindow[i].envelope;

        if(cabWindow[i].peak > maxPeak){
            maxPeak = cabWindow[i].peak;
        }

        if(cabWindow[i].rms > 0.00035f ||
           cabWindow[i].peak > 0.00120f ||
           cabWindow[i].envelope > 0.00035f ||
           cabWindow[i].spectralFlux > 0.00075f){
            activeFrames++;
        }
    }

    float meanRms = rmsSum / (float)cabWindowSize;
    float meanFlux = fluxSum / (float)cabWindowSize;
    float meanEnv = envSum / (float)cabWindowSize;

    return
        activeFrames >= 6 &&
        (
            maxPeak > 0.00120f ||
            meanRms > 0.00035f ||
            meanEnv > 0.00035f ||
            meanFlux > 0.00075f
        );
}

void StoreCABTrainingWindow(int label){
    if(label < 0 || label >= 4){
        return;
    }

    if(cabWindowCount < cabWindowSize){
        hw.seed.PrintLine("CAB waiting for window Count:%d/%d", cabWindowCount, cabWindowSize);
        return;
    }

    int activeFrames = 0;
    float maxPeak = 0.0f;
    float rmsSum = 0.0f;
    float fluxSum = 0.0f;
    float envSum = 0.0f;

    for(int i = 0; i < cabWindowSize; i++){
        rmsSum += cabWindow[i].rms;
        fluxSum += cabWindow[i].spectralFlux;
        envSum += cabWindow[i].envelope;

        if(cabWindow[i].peak > maxPeak){
            maxPeak = cabWindow[i].peak;
        }

        if(cabWindow[i].rms > 0.00035f ||
           cabWindow[i].peak > 0.00120f ||
           cabWindow[i].envelope > 0.00035f ||
           cabWindow[i].spectralFlux > 0.00075f){
            activeFrames++;
        }
    }

    float meanRms = rmsSum / (float)cabWindowSize;
    float meanFlux = fluxSum / (float)cabWindowSize;
    float meanEnv = envSum / (float)cabWindowSize;

    if(!CABWindowHasSignal()){
        hw.seed.PrintLine("CAB reject quiet AF:%d Peak:%d RMS:%d Env:%d Flux:%d",
            activeFrames,
            (int)(maxPeak * 1000000.0f),
            (int)(meanRms * 1000000.0f),
            (int)(meanEnv * 1000000.0f),
            (int)(meanFlux * 1000000.0f));
        return;
    }

    int slot = cabTrainIndex[label];
    int oldest = (cabWindowIndex - cabWindowSize + cabWindowSize) % cabWindowSize;

    for(int t = 0; t < cabWindowSize; t++){
        int src = (oldest + t) % cabWindowSize;
        cabTrainBuffer[label][slot][t] = cabWindow[src];
    }

    cabTrainIndex[label] = (cabTrainIndex[label] + 1) % cabExamplesPerClass;

    if(cabTrainCount[label] < cabExamplesPerClass){
        cabTrainCount[label]++;
    }

    hw.seed.PrintLine("Stored CAB window Label:%d Count:%d AF:%d Peak:%d RMS:%d Env:%d Flux:%d",
        label,
        cabTrainCount[label],
        activeFrames,
        (int)(maxPeak * 1000000.0f),
        (int)(meanRms * 1000000.0f),
        (int)(meanEnv * 1000000.0f),
        (int)(meanFlux * 1000000.0f));
}

void ResetPhrase(){
    phraseActive = false;
    phraseStartMs = 0;
    lastActiveMs = 0;
    phraseFrames = 0;
    lastPhraseReady = false;
    std::memset(&phraseAccum, 0, sizeof(AudioFeatures));

    phraseMaxRms = 0.0f;
    phraseMinRms = 1.0f;
    phraseMaxCentroid = 0.0f;
    phraseMinCentroid = 1000000.0f;
    phraseMaxPeak = 0.0f;
    phraseAttackCount = 0;
    lastPhraseAttackMs = 0;
    firstPhraseAttackMs = 0;
    prevPhraseAttackMs = 0;
    phraseAttackGapSumMs = 0.0f;
    phraseAttackGapCount = 0;
}

void AccumulatePhrase(const AudioFeatures& f, uint32_t nowMs){
    phraseAccum.rms += f.rms;
    phraseAccum.peak += f.peak;
    phraseAccum.zcr += f.zcr;
    phraseAccum.spectralCentroid += f.spectralCentroid;
    phraseAccum.spectralFlux += f.spectralFlux;
    phraseAccum.rmsDelta += f.rmsDelta;
    phraseAccum.envelope += f.envelope;
    phraseAccum.envelopeDelta += f.envelopeDelta;
    phraseAccum.rmsVariance += f.rmsVariance;
    phraseAccum.centroidVariance += f.centroidVariance;
    phraseAccum.onsetCount += f.onsetCount;
    phraseAccum.timeSinceLastOnset += f.timeSinceLastOnset;
    phraseFrames++;

    if(f.rms > phraseMaxRms) phraseMaxRms = f.rms;
    if(f.rms < phraseMinRms) phraseMinRms = f.rms;
    if(f.spectralCentroid > phraseMaxCentroid) phraseMaxCentroid = f.spectralCentroid;
    if(f.spectralCentroid < phraseMinCentroid) phraseMinCentroid = f.spectralCentroid;
    if(f.peak > phraseMaxPeak) phraseMaxPeak = f.peak;

    bool strongAttack = f.peak > 0.0075f && (f.spectralFlux > 0.035f || f.envelopeDelta > 0.012f || f.rmsDelta > 0.010f);

    bool cooldownExpired =
        lastPhraseAttackMs == 0 ||
        (nowMs - lastPhraseAttackMs) > phraseAttackCooldownMs;

    if(strongAttack && cooldownExpired){
        if(phraseAttackCount == 0){
            firstPhraseAttackMs = nowMs;
        }
        else{
            phraseAttackGapSumMs += (float)(nowMs - prevPhraseAttackMs);
            phraseAttackGapCount++;
        }

        phraseAttackCount++;
        prevPhraseAttackMs = nowMs;
        lastPhraseAttackMs = nowMs;
    }
}

void FinalisePhrase(AudioFeatures& out){
    std::memset(&out, 0, sizeof(AudioFeatures));

    if(phraseFrames <= 0){
        return;
    }

    float inv = 1.0f / (float)phraseFrames;

    out.rms = phraseAccum.rms * inv;
    out.peak = phraseMaxPeak;
    out.zcr = phraseAccum.zcr * inv;
    out.spectralCentroid = phraseAccum.spectralCentroid * inv;
    out.spectralFlux = phraseAccum.spectralFlux * inv;
    out.envelope = phraseAccum.envelope * inv;
    out.rmsVariance = phraseMaxRms - phraseMinRms;
    out.centroidVariance = phraseMaxCentroid - phraseMinCentroid;

    uint32_t phraseEndMs = lastActiveMs;

    if(phraseEndMs < phraseStartMs){
        phraseEndMs = phraseStartMs;
    }

    float phraseDurationSec = (float)(phraseEndMs - phraseStartMs) / 1000.0f;

    if(phraseDurationSec < 0.001f){
        phraseDurationSec = 0.001f;
    }

    float attacksPerSecond = (float)phraseAttackCount / phraseDurationSec;

    uint32_t gapFromLastAttackMs = 0;

    if(lastPhraseAttackMs > 0 && phraseEndMs >= lastPhraseAttackMs){
        gapFromLastAttackMs = phraseEndMs - lastPhraseAttackMs;
    }
    else{
        gapFromLastAttackMs = phraseEndMs - phraseStartMs;
    }

    float tailGapSec = (float)gapFromLastAttackMs / 1000.0f;
    float sustainRatio = tailGapSec / phraseDurationSec;

    if(sustainRatio > 1.0f){
        sustainRatio = 1.0f;
    }

    out.onsetCount = (float)phraseAttackCount;
    out.rmsDelta = attacksPerSecond;
    out.envelopeDelta = sustainRatio;
    out.timeSinceLastOnset = tailGapSec;
}

bool IsPhraseSignalActive(const AudioFeatures& f){
    return
        f.rms > 0.0030f ||
        f.peak > 0.0060f ||
        f.envelope > 0.0030f;
}

bool ShouldAcceptTrainingPhrase(int label, const AudioFeatures& phraseFeatures, uint32_t nowMs){
    (void)nowMs;

    if(label < 0 || label >= 4){
        return false;
    }

    if(bufferCount[label] >= targetPhrasesPerClass){
        hw.seed.PrintLine("Reject reason: class full");
        return false;
    }

    bool hasEnoughSignal =
        phraseFeatures.rms > 0.0008f ||
        phraseFeatures.peak > 0.0020f ||
        phraseFeatures.envelope > 0.0008f;

    bool hasUsefulMovement =
        phraseFeatures.spectralFlux > 0.003f ||
        phraseFeatures.onsetCount >= 1.0f ||
        phraseFeatures.rmsVariance > 0.0002f ||
        phraseFeatures.centroidVariance > 20.0f;

    bool ok = hasEnoughSignal && hasUsefulMovement;

    if(!ok){
        hw.seed.PrintLine("Reject reason: weak phrase Label:%d RMS:%d Peak:%d Flux:%d On:%d",
            label,
            (int)(phraseFeatures.rms * 1000.0f),
            (int)(phraseFeatures.peak * 1000.0f),
            (int)(phraseFeatures.spectralFlux * 1000.0f),
            (int)(phraseFeatures.onsetCount * 1000.0f));
    }

    return ok;
}

void StorePhraseExample(int label, const AudioFeatures& phraseFeatures, uint32_t nowMs){
    featureBuffer[label][bufferIndex[label]] = phraseFeatures;

    bufferIndex[label] = (bufferIndex[label] + 1) % perClassBufferCapacity;

    if(bufferCount[label] < perClassBufferCapacity){
        bufferCount[label]++;
    }

    lastAcceptedPhraseMs[label] = nowMs;
}

void ExtractRawArray(const AudioFeatures& f, float raw[12]){
    raw[0] = f.rms;
    raw[1] = f.peak;
    raw[2] = f.zcr;
    raw[3] = f.spectralCentroid;
    raw[4] = f.spectralFlux;
    raw[5] = f.rmsDelta;
    raw[6] = f.envelope;
    raw[7] = f.envelopeDelta;
    raw[8] = f.rmsVariance;
    raw[9] = f.centroidVariance;
    raw[10] = f.onsetCount;
    raw[11] = f.timeSinceLastOnset;
}

bool ComputePhraseDatasetNormalisation(){
    for(int i = 0; i < 12; i++){
        datasetMean[i] = 0.0f;
        datasetStd[i] = 0.0f;
    }

    int totalCount = 0;

    for(int classId = 0; classId < 4; classId++){
        for(int i = 0; i < bufferCount[classId]; i++){
            float raw[12];
            ExtractRawArray(featureBuffer[classId][i], raw);

            for(int j = 0; j < 12; j++){
                datasetMean[j] += raw[j];
            }

            totalCount++;
        }
    }

    if(totalCount <= 0){
        return false;
    }

    float inv = 1.0f / (float)totalCount;

    for(int j = 0; j < 12; j++){
        datasetMean[j] *= inv;
    }

    for(int classId = 0; classId < 4; classId++){
        for(int i = 0; i < bufferCount[classId]; i++){
            float raw[12];
            ExtractRawArray(featureBuffer[classId][i], raw);

            for(int j = 0; j < 12; j++){
                float d = raw[j] - datasetMean[j];
                datasetStd[j] += d * d;
            }
        }
    }

    for(int j = 0; j < 12; j++){
        datasetStd[j] = sqrtf(datasetStd[j] * inv);

        if(datasetStd[j] < 1e-6f){
            datasetStd[j] = 1.0f;
        }
    }

    return true;
}

bool ComputeCABDatasetNormalisation(){
    for(int i = 0; i < 12; i++){
        datasetMean[i] = 0.0f;
        datasetStd[i] = 0.0f;
    }

    int totalCount = 0;

    for(int classId = 0; classId < 4; classId++){
        for(int i = 0; i < cabTrainCount[classId]; i++){
            for(int t = 0; t < cabWindowSize; t++){
                float raw[12];
                ExtractRawArray(cabTrainBuffer[classId][i][t], raw);

                for(int j = 0; j < 12; j++){
                    datasetMean[j] += raw[j];
                }

                totalCount++;
            }
        }
    }

    if(totalCount <= 0){
        return false;
    }

    float inv = 1.0f / (float)totalCount;

    for(int j = 0; j < 12; j++){
        datasetMean[j] *= inv;
    }

    for(int classId = 0; classId < 4; classId++){
        for(int i = 0; i < cabTrainCount[classId]; i++){
            for(int t = 0; t < cabWindowSize; t++){
                float raw[12];
                ExtractRawArray(cabTrainBuffer[classId][i][t], raw);

                for(int j = 0; j < 12; j++){
                    float d = raw[j] - datasetMean[j];
                    datasetStd[j] += d * d;
                }
            }
        }
    }

    for(int j = 0; j < 12; j++){
        datasetStd[j] = sqrtf(datasetStd[j] * inv);

        if(datasetStd[j] < 1e-6f){
            datasetStd[j] = 1.0f;
        }
    }

    return true;
}

int CountReadyClasses(){
    int readyClasses = 0;

    for(int i = 0; i < 4; i++){
        if(bufferCount[i] >= 1){
            readyClasses++;
        }
    }

    return readyClasses;
}

int CountReadyCABClasses(){
    int readyClasses = 0;

    for(int i = 0; i < 4; i++){
        if(cabTrainCount[i] >= minCABExamplesPerClassForTraining){
            readyClasses++;
        }
    }

    return readyClasses;
}

bool TrainNetworkFromBuffers(){
    if(CountReadyClasses() < 2){
        return false;
    }

    if(!ComputePhraseDatasetNormalisation()){
        return false;
    }

    neuralnet.Init();
    neuralnet.Normalisation(datasetMean, datasetStd);

    static constexpr int epochs = 90;
    static constexpr float eta = 0.0035f;

    for(int epoch = 0; epoch < epochs; epoch++){
        for(int classId = 0; classId < 4; classId++){
            for(int i = 0; i < bufferCount[classId]; i++){
                neuralnet.Train(featureBuffer[classId][i], classId, eta);
            }
        }
    }

    neuralnet.SaveState(savedMapping);
    return true;
}

bool TrainMiniCABFromBuffers(){
    if(CountReadyCABClasses() < 2){
        return false;
    }

    if(!ComputeCABDatasetNormalisation()){
        return false;
    }

    minicab.Init();
    minicab.SetNormalisation(datasetMean, datasetStd);
    minicab.Reset();

    static constexpr int epochs = 300;
    static constexpr float eta = 0.0010f;
    static constexpr int maxTrainItems = 4 * cabExamplesPerClass;

    int maxCount = 0;

    for(int classId = 0; classId < 4; classId++){
        if(cabTrainCount[classId] > maxCount){
            maxCount = cabTrainCount[classId];
        }
    }

    if(maxCount <= 0){
        return false;
    }

    int trainClass[maxTrainItems];
    int trainIndex[maxTrainItems];

    for(int epoch = 0; epoch < epochs; epoch++){
        int trainCount = 0;

        for(int n = 0; n < maxCount; n++){
            for(int classId = 0; classId < 4; classId++){
                if(cabTrainCount[classId] > 0 && trainCount < maxTrainItems){
                    trainClass[trainCount] = classId;
                    trainIndex[trainCount] = (n + epoch) % cabTrainCount[classId];
                    trainCount++;
                }
            }
        }

        for(int i = trainCount - 1; i > 0; i--){
            int j = rand() % (i + 1);

            int tmpClass = trainClass[i];
            int tmpIndex = trainIndex[i];

            trainClass[i] = trainClass[j];
            trainIndex[i] = trainIndex[j];

            trainClass[j] = tmpClass;
            trainIndex[j] = tmpIndex;
        }

        for(int n = 0; n < trainCount; n++){
            int classId = trainClass[n];
            int exampleIndex = trainIndex[n];

            minicab.Reset();

            for(int t = 0; t < cabWindowSize; t++){
                minicab.PushFrame(cabTrainBuffer[classId][exampleIndex][t]);
            }

            minicab.TrainFromCurrentSequence(classId, eta);
        }
    }

    minicab.Reset();
    return true;
}

bool IsTransitionOnlyPhrase(const AudioFeatures& phraseFeatures){
    bool pureTail =
        phraseFeatures.onsetCount < 0.5f &&
        phraseFeatures.envelopeDelta > 0.90f &&
        phraseFeatures.timeSinceLastOnset > 0.35f;

    bool freshTransientOnly =
        phraseFeatures.onsetCount <= 1.0f &&
        phraseFeatures.envelopeDelta < 0.06f &&
        phraseFeatures.timeSinceLastOnset < 0.04f;

    return pureTail || freshTransientOnly;
}

void ResetPredictionState(){
    predictedCandidate = currentPresetIndex;
    predictedStableClass = currentPresetIndex;
    predictedHoldCount = 0;
    stableClassAge = 0;
    candidateClassAge = 0;

    for(int i = 0; i < 4; i++){
        classScores[i] = 0.0f;
        classDistances[i] = 9999.0f;
        smoothedClassScores[i] = 0.25f;
        nnOutput.scores[i] = 0.0f;
        cabOutput.scores[i] = 0.0f;
    }

    nnOutput.predictedClass = currentPresetIndex;
    cabOutput.predictedClass = currentPresetIndex;
}

void UpdatePredictionState(bool allowImmediateSwitch, const AudioFeatures& phraseFeatures){
    (void)allowImmediateSwitch;

    if(currentNetworkMode != NETWORK_MINICAB && IsTransitionOnlyPhrase(phraseFeatures)){
        return;
    }

    int rawBestClass = nnOutput.predictedClass;
    float rawBestScore = nnOutput.scores[rawBestClass];
    float rawSecondScore = 0.0f;

    for(int i = 0; i < 4; i++){
        if(i != rawBestClass && nnOutput.scores[i] > rawSecondScore){
            rawSecondScore = nnOutput.scores[i];
        }

        float smoothAmount = currentNetworkMode == NETWORK_MINICAB ? 0.45f : 0.60f;
        float newAmount = 1.0f - smoothAmount;

        smoothedClassScores[i] = smoothAmount * smoothedClassScores[i] + newAmount * nnOutput.scores[i];
        classScores[i] = smoothedClassScores[i];
        classDistances[i] = 1.0f - smoothedClassScores[i];
    }

    int bestClass = 0;
    float bestScore = smoothedClassScores[0];
    float secondScore = 0.0f;

    for(int i = 1; i < 4; i++){
        float s = smoothedClassScores[i];

        if(s > bestScore){
            secondScore = bestScore;
            bestScore = s;
            bestClass = i;
        }
        else if(s > secondScore){
            secondScore = s;
        }
    }

    float rawMargin = rawBestScore - rawSecondScore;
    float margin = bestScore - secondScore;

    bool confident;

    if(currentNetworkMode == NETWORK_MINICAB){
        confident =
            rawBestScore > 0.30f &&
            rawMargin > 0.015f &&
            bestScore > 0.28f &&
            margin > 0.010f;
    }
    else{
        confident =
            rawBestScore > 0.48f &&
            rawMargin > 0.10f &&
            bestScore > 0.36f &&
            margin > 0.06f;
    }

    if(!confident){
        stableClassAge++;
        return;
    }

    if(bestClass == predictedCandidate){
        predictedHoldCount++;
        candidateClassAge++;
    }
    else{
        predictedCandidate = bestClass;
        predictedHoldCount = 1;
        candidateClassAge = 1;
    }

    int requiredHold = currentNetworkMode == NETWORK_MINICAB ? 2 : 2;

    if(bestScore > 0.60f && margin > 0.18f){
        requiredHold = 1;
    }

    if(predictedHoldCount >= requiredHold){
        predictedStableClass = predictedCandidate;
        stableClassAge = 0;
    }
    else{
        stableClassAge++;
    }
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
        case CLIP_BYPASS: return x;
        case CLIP_SOFT: return MySoftClip(x);
        case CLIP_HARD: return HardClip(x, threshold);
        case CLIP_ASYM: return AsymClip(x);
        case CLIP_FUZZ: return FuzzClip(x);
        default: return x;
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

void ApplyPresetIndex(int presetIndex){
    if(presetIndex < 0){
        presetIndex = 0;
    }

    if(presetIndex >= noNumTonePresets){
        presetIndex = noNumTonePresets - 1;
    }

    currentPresetIndex = presetIndex;
    currEffectMode = static_cast<WhichEffect>(presetIndex);
    SetTonePreset(tonePresets[presetIndex]);
}

void ClearDelayMemory(){
    std::memset(delayBuffer, 0, sizeof(delayBuffer));
    writeIndex = 0;
    toneStateL = 0.0f;
    ambientToneState = 0.0f;
    reverbToneState = 0.0f;
}

void ApplyLearnedToneMapping(){
    float weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int effectClass = currentLabel;

    if(predictionMode && CurrentNetworkHasSavedMapping()){
        effectClass = predictedStableClass;

        if(effectClass < 0){
            effectClass = 0;
        }

        if(effectClass >= 4){
            effectClass = 3;
        }

        float scoreSum = 0.0f;

        for(int i = 0; i < 4; i++){
            weights[i] = classScores[i];

            if(weights[i] < 0.0f){
                weights[i] = 0.0f;
            }

            scoreSum += weights[i];
        }

        if(scoreSum < 0.0001f){
            for(int i = 0; i < 4; i++){
                weights[i] = 0.0f;
            }

            weights[effectClass] = 1.0f;
        }
    }
    else{
        effectClass = currentLabel;

        if(effectClass < 0){
            effectClass = 0;
        }

        if(effectClass >= 4){
            effectClass = 3;
        }

        weights[effectClass] = 1.0f;
    }

    float weightSum = 0.0f;

    for(int i = 0; i < 4; i++){
        weightSum += weights[i];
    }

    if(weightSum < 0.0001f){
        for(int i = 0; i < 4; i++){
            weights[i] = 0.0f;
        }

        weights[effectClass] = 1.0f;
        weightSum = 1.0f;
    }

    for(int i = 0; i < 4; i++){
        weights[i] /= weightSum;
    }

    currentPresetIndex = effectClass;
    currEffectMode = static_cast<WhichEffect>(effectClass);

    if(currEffectMode != lastAppliedEffectMode){
        ClearDelayMemory();
        lastAppliedEffectMode = currEffectMode;
    }

    currentClipMode = tonePresets[effectClass].clipMode;

    float basePreamp = 0.0f;
    float baseDrive = 0.0f;
    float baseTone = 0.0f;
    float baseMix = 0.0f;
    float baseLevel = 0.0f;
    float baseThreshold = 0.0f;

    for(int i = 0; i < 4; i++){
        basePreamp += weights[i] * tonePresets[i].preamp;
        baseDrive += weights[i] * tonePresets[i].drive;
        baseTone += weights[i] * tonePresets[i].tone;
        baseMix += weights[i] * tonePresets[i].mix;
        baseLevel += weights[i] * tonePresets[i].level;
        baseThreshold += weights[i] * tonePresets[i].clipThreshold;
    }

    float energy = Clamp(smoothedFeatures.rms, 0.0f, 0.08f) / 0.08f;
    float peak = Clamp(smoothedFeatures.peak, 0.0f, 0.18f) / 0.18f;
    float flux = Clamp(smoothedFeatures.spectralFlux, 0.0f, 0.12f) / 0.12f;
    float movement = Clamp(smoothedFeatures.rmsDelta + smoothedFeatures.envelopeDelta + smoothedFeatures.spectralFlux * 2.0f, 0.0f, 1.0f);

    float centroid = Clamp(smoothedFeatures.spectralCentroid, 200.0f, 4200.0f);
    float brightness = (centroid - 200.0f) / 4000.0f;

    float sustain = Clamp(smoothedFeatures.envelopeDelta, 0.0f, 1.0f);
    float gap = Clamp(smoothedFeatures.timeSinceLastOnset, 0.0f, 1.5f) / 1.5f;

    targetPreamp = Clamp(basePreamp + 0.30f * energy, 0.5f, 7.0f);
    targetDrive = Clamp(baseDrive + 1.25f * energy + 0.55f * peak, 0.5f, 8.0f);
    targetTone = Clamp(baseTone + 0.24f * brightness, 0.05f, 0.85f);
    targetMix = Clamp(baseMix + 0.08f * flux, 0.05f, 1.0f);
    targetLevel = Clamp(baseLevel - 0.12f * energy, 0.35f, 1.8f);
    targetClipThreshold = Clamp(baseThreshold - 0.10f * energy, 0.20f, 2.0f);

    if(currEffectMode == MODE_CHORUS){
        float nextChorusMix = Clamp(0.22f + 0.35f * targetMix + 0.04f * flux, 0.18f, 0.55f);
        float nextRateHz = Clamp(0.18f + 0.45f * movement, 0.15f, 0.75f);
        float nextDepthMs = Clamp(0.8f + 2.0f * flux + 0.6f * brightness, 0.6f, 3.2f);

        chorusMix = 0.92f * chorusMix + 0.08f * nextChorusMix;
        rateHz = 0.92f * rateHz + 0.08f * nextRateHz;
        depthMs = 0.92f * depthMs + 0.08f * nextDepthMs;

        depth = sampleRate * depthMs / 1000.0f;
        phaseIncrement = 2.0f * 3.14159265359f * rateHz / sampleRate;
    }
    else if(currEffectMode == MODE_AMBIENT){
        float nextAmbientMix = Clamp(0.40f + 0.50f * targetMix + 0.08f * sustain + 0.05f * gap, 0.35f, 0.98f);
        float nextAmbientFeedback = Clamp(0.70f + 0.18f * targetMix + 0.05f * sustain, 0.65f, 0.96f);
        float nextAmbientLevel = Clamp(1.4f + 3.2f * targetMix + 0.6f * energy, 1.4f, 5.0f);

        ambientMix = 0.90f * ambientMix + 0.10f * nextAmbientMix;
        ambientFeedback = 0.90f * ambientFeedback + 0.10f * nextAmbientFeedback;
        ambientLevel = 0.90f * ambientLevel + 0.10f * nextAmbientLevel;
    }
    else if(currEffectMode == MODE_REVERB){
        float nextReverbMix = Clamp(0.30f + 0.55f * targetMix + 0.08f * sustain, 0.25f, 0.94f);
        float nextReverbFeedback = Clamp(0.64f + 0.20f * targetMix + 0.06f * sustain, 0.60f, 0.94f);
        float nextReverbLevel = Clamp(1.4f + 2.6f * targetMix + 0.5f * energy, 1.4f, 4.5f);

        reverbMix = 0.90f * reverbMix + 0.10f * nextReverbMix;
        reverbFeedback = 0.90f * reverbFeedback + 0.10f * nextReverbFeedback;
        reverbLevel = 0.90f * reverbLevel + 0.10f * nextReverbLevel;
    }
}

float ProcessTone(float x){
    float pre = x * smoothedPreamp;
    float stage1 = ApplyClipper(pre * smoothedDrive, currentClipMode, smoothedClipThreshold);
    float stage2 = tanhf(stage1 * 1.35f);
    float mixed = (1.0f - smoothedMix) * x + smoothedMix * stage2;
    toneStateL = toneStateL + smoothedTone * (mixed - toneStateL);
    float y = toneStateL * smoothedLevel;

    return Clamp(y, -1.0f, 1.0f);
}

float ReadDelay(float delay){
    float readPos = (float)writeIndex - delay;

    while(readPos < 0.0f) readPos += bufferSize;
    while(readPos >= bufferSize) readPos -= bufferSize;

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

            float wetDry = (1.0f - chorusMix) * x + chorusMix * delayed;
            y = Clamp(wetDry * smoothedLevel * 1.55f, -1.0f, 1.0f);

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
    float fb = 0.45f * x + ambientFeedback * softened;

    if(fb > 1.0f) fb = 1.0f;
    if(fb < -1.0f) fb = -1.0f;

    delayBuffer[writeIndex] = fb;
    writeIndex = (writeIndex + 1) % bufferSize;

    ambientPhase += ambientPhaseIncrement;

    if(ambientPhase >= 6.28318530718f){
        ambientPhase -= 6.28318530718f;
    }

    float y = ((1.0f - ambientMix) * x + ambientMix * softened) * ambientLevel * smoothedLevel;
    y = Clamp(y, -1.0f, 1.0f);

    return y;
}

float reverbProcessor(float x){
    float d1 = ReadDelay(reverbDelay1);
    float d2 = ReadDelay(reverbDelay2);
    float d3 = ReadDelay(reverbDelay3);

    float echoes = 0.55f * d1 + 0.30f * d2 + 0.15f * d3;
    reverbToneState = reverbToneState + 0.03f * (echoes - reverbToneState);

    float fb = 0.55f * x + reverbFeedback * reverbToneState;

    if(fb > 1.0f) fb = 1.0f;
    if(fb < -1.0f) fb = -1.0f;

    delayBuffer[writeIndex] = fb;
    writeIndex = (writeIndex + 1) % bufferSize;

    float y = ((1.0f - reverbMix) * x + reverbMix * echoes) * reverbLevel * smoothedLevel;
    y = Clamp(y, -1.0f, 1.0f);

    return y;
}

void AudioCallBack(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size){
    chorusSampler(in, out, size);
}

float GenerateSignal(){
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

bool PredictCABWindow(){
    if(cabWindowCount < cabWindowSize){
        return false;
    }

    if(!CABWindowHasSignal()){
        return false;
    }

    minicab.Reset();

    int oldest = (cabWindowIndex - cabWindowSize + cabWindowSize) % cabWindowSize;

    for(int t = 0; t < cabWindowSize; t++){
        int src = (oldest + t) % cabWindowSize;
        minicab.PushFrame(cabWindow[src]);
    }

    cabOutput = minicab.PredictSequence();

    for(int i = 0; i < 4; i++){
        nnOutput.scores[i] = cabOutput.scores[i];
        classScores[i] = cabOutput.scores[i];
        classDistances[i] = 1.0f - cabOutput.scores[i];
    }

    nnOutput.predictedClass = cabOutput.predictedClass;
    return true;
}

void ClassifyCurrentNetwork(const AudioFeatures& phraseFeatures){
    if(currentNetworkMode == NETWORK_NEURALNET){
        nnOutput = neuralnet.Predict(phraseFeatures);

        for(int i = 0; i < 4; i++){
            classScores[i] = nnOutput.scores[i];
            classDistances[i] = 1.0f - nnOutput.scores[i];
        }
    }
    else{
        PredictCABWindow();
    }
}

int main(void){
    hw.Init();
    hw.seed.StartLog();
    System::Delay(1000);

    hw.seed.PrintLine("Boot 1: hw init done");

    ScreenInit();
    hw.seed.PrintLine("Boot 2: screen init done");

    ScreenFillBlack();
    hw.seed.PrintLine("Boot 3: screen black done");

    hw.SetAudioBlockSize(48);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    extractor.Init(256, 48000.0f);
    neuralnet.Init();
    minicab.Init();
    minicab.SetNormalisation(datasetMean, datasetStd);

    hw.seed.PrintLine("Boot 4: models init done");

    ApplyPresetIndex(currentPresetIndex);
    ResetPhrase();
    ResetPredictionState();
    ResetCABWindow();

    hw.seed.PrintLine("Starting...");
    hw.seed.PrintLine("Preset: %s", tonePresets[currentPresetIndex].toneName);

    hw.StartAudio(AudioCallBack);
    hw.seed.PrintLine("Audio started");

    static constexpr int MENU_PRESET = 0;
    static constexpr int MENU_NETWORK = 1;
    static constexpr int MENU_TRAINING = 2;
    static constexpr int MENU_SAVE = 3;
    static constexpr int MENU_PREDICTION = 4;
    static constexpr int MENU_COUNT = 5;

    int menuIndex = 0;
    int modeIndex = currentPresetIndex;

    uint32_t lastPrintTime = 0;
    uint32_t lastScreenUpdate = 0;

    int lastMenuIndex = -1;
    int lastPreset = -1;
    int lastTraining = -1;
    int lastPredictionMode = -1;
    int lastNetworkMode = -1;
    int lastSaved = -1;
    int lastB0 = -1;
    int lastB1 = -1;
    int lastB2 = -1;
    int lastB3 = -1;

    while(1) {
        hw.ProcessAnalogControls();
        hw.ProcessDigitalControls();

        int inc = hw.encoder.Increment();

        if(inc > 0){
            menuIndex++;

            if(menuIndex >= MENU_COUNT){
                menuIndex = 0;
            }

            ForceScreenRedraw(
                lastMenuIndex,
                lastPreset,
                lastTraining,
                lastPredictionMode,
                lastNetworkMode,
                lastSaved,
                lastB0,
                lastB1,
                lastB2,
                lastB3
            );

            hw.seed.PrintLine("Menu:%d", menuIndex);
        }
        else if(inc < 0){
            menuIndex--;

            if(menuIndex < 0){
                menuIndex = MENU_COUNT - 1;
            }

            ForceScreenRedraw(
                lastMenuIndex,
                lastPreset,
                lastTraining,
                lastPredictionMode,
                lastNetworkMode,
                lastSaved,
                lastB0,
                lastB1,
                lastB2,
                lastB3
            );

            hw.seed.PrintLine("Menu:%d", menuIndex);
        }

        if(hw.encoder.RisingEdge()){
            if(menuIndex == MENU_PRESET){
                predictionMode = false;

                modeIndex = (modeIndex + 1) % 4;
                currentLabel = modeIndex;
                ApplyPresetIndex(currentLabel);

                ResetPredictionState();
                ResetPhrase();
                ResetCABWindow();

                hw.seed.PrintLine("Manual preset: %s", tonePresets[currentLabel].toneName);
            }
            else if(menuIndex == MENU_NETWORK){
                if(currentNetworkMode == NETWORK_NEURALNET){
                    currentNetworkMode = NETWORK_MINICAB;
                    minicab.SetNormalisation(datasetMean, datasetStd);
                    minicab.Reset();

                    ResetPredictionState();
                    ResetPhrase();
                    ResetCABWindow();

                    hw.seed.PrintLine("Network: MiniCAB");
                }
                else{
                    currentNetworkMode = NETWORK_NEURALNET;
                    ResetPredictionState();
                    ResetPhrase();
                    ResetCABWindow();

                    hw.seed.PrintLine("Network: NeuralNet");
                }
            }
            else if(menuIndex == MENU_TRAINING){
                isTraining = !isTraining;

                if(isTraining){
                    predictionMode = false;
                    ResetPredictionState();
                    ResetPhrase();
                    ResetCABWindow();
                    lastCABStoreMs = System::GetNow();

                    hw.seed.PrintLine("Training ON Label:%d Preset:%s",
                        currentLabel,
                        tonePresets[currentLabel].toneName);
                }
                else{
                    ResetPhrase();
                    ResetCABWindow();
                    hw.seed.PrintLine("Training OFF");
                }
            }
            else if(menuIndex == MENU_SAVE){
                int readyClasses = currentNetworkMode == NETWORK_MINICAB ? CountReadyCABClasses() : CountReadyClasses();

                if(readyClasses < 2){
                    hw.seed.PrintLine("Need 2 trained classes before saving");
                }
                else{
                    bool trainedOk = false;

                    if(currentNetworkMode == NETWORK_NEURALNET){
                        trainedOk = TrainNetworkFromBuffers();

                        if(trainedOk){
                            hasSavedNeuralNetMapping = true;
                            predictionMode = true;
                            isTraining = false;
                            neuralnet.LoadState(savedMapping);
                            ResetPhrase();
                            ResetPredictionState();

                            hw.seed.PrintLine("Saved NeuralNet mapping and entered prediction mode");
                        }
                        else{
                            hw.seed.PrintLine("NeuralNet training failed");
                        }
                    }
                    else{
                        trainedOk = TrainMiniCABFromBuffers();

                        if(trainedOk){
                            hasSavedCABMapping = true;
                            predictionMode = true;
                            isTraining = false;
                            minicab.SetNormalisation(datasetMean, datasetStd);
                            minicab.Reset();

                            ResetPredictionState();
                            ResetPhrase();
                            ResetCABWindow();

                            hw.seed.PrintLine("Trained MiniCAB and entered prediction mode");
                        }
                        else{
                            hw.seed.PrintLine("MiniCAB training failed");
                        }
                    }
                }
            }
            else if(menuIndex == MENU_PREDICTION){
                if(CurrentNetworkHasSavedMapping()){
                    predictionMode = !predictionMode;

                    if(predictionMode){
                        isTraining = false;
                    }

                    ResetPredictionState();
                    ResetPhrase();
                    ResetCABWindow();

                    hw.seed.PrintLine("Prediction %s", predictionMode ? "ON" : "OFF");
                }
                else{
                    hw.seed.PrintLine("No saved mapping yet");
                }
            }

            ForceScreenRedraw(
                lastMenuIndex,
                lastPreset,
                lastTraining,
                lastPredictionMode,
                lastNetworkMode,
                lastSaved,
                lastB0,
                lastB1,
                lastB2,
                lastB3
            );
        }

        currentLabel = modeIndex;

        if(extractor.ProcessFrame(latestFeatures)) {
            SmoothFeatures(latestFeatures, smoothedFeatures);

            smoothedFeatures.rms = Clamp(smoothedFeatures.rms, 0.0f, 1.0f);
            smoothedFeatures.peak = Clamp(smoothedFeatures.peak, 0.0f, 1.0f);
            smoothedFeatures.zcr = Clamp(smoothedFeatures.zcr, 0.0f, 1.0f);
            smoothedFeatures.spectralCentroid = Clamp(smoothedFeatures.spectralCentroid, 0.0f, 5000.0f);
            smoothedFeatures.spectralFlux = Clamp(smoothedFeatures.spectralFlux, 0.0f, 5.0f);
            smoothedFeatures.rmsDelta = Clamp(smoothedFeatures.rmsDelta, 0.0f, 10.0f);
            smoothedFeatures.envelope = Clamp(smoothedFeatures.envelope, 0.0f, 1.0f);
            smoothedFeatures.envelopeDelta = Clamp(smoothedFeatures.envelopeDelta, 0.0f, 1.0f);
            smoothedFeatures.rmsVariance = Clamp(smoothedFeatures.rmsVariance, 0.0f, 1.0f);
            smoothedFeatures.centroidVariance = Clamp(smoothedFeatures.centroidVariance, 0.0f, 500000.0f);
            smoothedFeatures.onsetCount = Clamp(smoothedFeatures.onsetCount, 0.0f, 8.0f);
            smoothedFeatures.timeSinceLastOnset = Clamp(smoothedFeatures.timeSinceLastOnset, 0.0f, 2.0f);

            PushCABWindowFrame(smoothedFeatures);

            uint32_t nowMs = System::GetNow();

            if(isTraining && currentNetworkMode == NETWORK_MINICAB){
                if(nowMs - lastCABStoreMs >= 500){
                    lastCABStoreMs = nowMs;
                    StoreCABTrainingWindow(currentLabel);
                }
            }

            if(predictionMode && CurrentNetworkHasSavedMapping() && currentNetworkMode == NETWORK_MINICAB){
                if(nowMs - lastCABPredictMs >= 250){
                    lastCABPredictMs = nowMs;

                    if(PredictCABWindow()){
                        UpdatePredictionState(true, smoothedFeatures);
                    }
                }
            }

            bool activeNow = IsPhraseSignalActive(smoothedFeatures);

            if(activeNow){
                if(!phraseActive){
                    phraseActive = true;
                    phraseStartMs = nowMs;
                    lastActiveMs = nowMs;
                    phraseFrames = 0;

                    std::memset(&phraseAccum, 0, sizeof(AudioFeatures));

                    phraseMaxRms = 0.0f;
                    phraseMinRms = 1.0f;
                    phraseMaxCentroid = 0.0f;
                    phraseMinCentroid = 1000000.0f;
                    phraseMaxPeak = 0.0f;
                    phraseAttackCount = 0;
                    lastPhraseAttackMs = 0;
                    firstPhraseAttackMs = 0;
                    prevPhraseAttackMs = 0;
                    phraseAttackGapSumMs = 0.0f;
                    phraseAttackGapCount = 0;
                }

                lastActiveMs = nowMs;
                AccumulatePhrase(smoothedFeatures, nowMs);

                if(predictionMode && CurrentNetworkHasSavedMapping() && currentNetworkMode == NETWORK_NEURALNET && phraseFrames >= minPhraseFrames){
                    AudioFeatures livePhraseFeatures;
                    FinalisePhrase(livePhraseFeatures);
                    ClassifyCurrentNetwork(livePhraseFeatures);
                    UpdatePredictionState(true, livePhraseFeatures);
                }
            }

            bool phraseEndedByGap =
                phraseActive &&
                !activeNow &&
                (nowMs - lastActiveMs) > phraseEndGapMs;

            bool phraseTimedOut =
                phraseActive &&
                (nowMs - phraseStartMs) > maxPhraseLengthMs;

            if(phraseEndedByGap || phraseTimedOut){
                AudioFeatures phraseFeatures;
                FinalisePhrase(phraseFeatures);
                lastPhraseFeatures = phraseFeatures;
                lastPhraseReady = true;

                if(phraseFrames >= minPhraseFrames){
                    if(isTraining && currentNetworkMode == NETWORK_NEURALNET){
                        bool acceptPhrase = ShouldAcceptTrainingPhrase(currentLabel, phraseFeatures, nowMs);

                        if(acceptPhrase){
                            StorePhraseExample(currentLabel, phraseFeatures, nowMs);

                            hw.seed.PrintLine("Accepted phrase Label:%d Frames:%d Count:%d Timeout:%d",
                                currentLabel,
                                phraseFrames,
                                bufferCount[currentLabel],
                                (int)phraseTimedOut);
                        }
                        else{
                            hw.seed.PrintLine("Rejected phrase Label:%d Frames:%d Count:%d Timeout:%d",
                                currentLabel,
                                phraseFrames,
                                bufferCount[currentLabel],
                                (int)phraseTimedOut);
                        }
                    }
                    else if(predictionMode && CurrentNetworkHasSavedMapping() && currentNetworkMode == NETWORK_NEURALNET){
                        ClassifyCurrentNetwork(phraseFeatures);
                        UpdatePredictionState(true, phraseFeatures);

                        hw.seed.PrintLine("Captured prediction phrase Pred:%d Stable:%d Frames:%d Timeout:%d",
                            nnOutput.predictedClass,
                            predictedStableClass,
                            phraseFrames,
                            (int)phraseTimedOut);
                    }
                }

                ResetPhrase();
            }

            ApplyLearnedToneMapping();

            smoothedDrive = 0.45f * smoothedDrive + 0.55f * targetDrive;
            smoothedTone = 0.45f * smoothedTone + 0.55f * targetTone;
            smoothedPreamp = 0.45f * smoothedPreamp + 0.55f * targetPreamp;
            smoothedMix = 0.45f * smoothedMix + 0.55f * targetMix;
            smoothedLevel = 0.45f * smoothedLevel + 0.55f * targetLevel;
            smoothedClipThreshold = 0.45f * smoothedClipThreshold + 0.55f * targetClipThreshold;
        }

        uint32_t nowScreen = System::GetNow();

        if(nowScreen - lastScreenUpdate >= 30){
            lastScreenUpdate = nowScreen;

            int displayB0 = currentNetworkMode == NETWORK_MINICAB ? cabTrainCount[0] : bufferCount[0];
            int displayB1 = currentNetworkMode == NETWORK_MINICAB ? cabTrainCount[1] : bufferCount[1];
            int displayB2 = currentNetworkMode == NETWORK_MINICAB ? cabTrainCount[2] : bufferCount[2];
            int displayB3 = currentNetworkMode == NETWORK_MINICAB ? cabTrainCount[3] : bufferCount[3];

            bool needRedraw = false;

            if(menuIndex != lastMenuIndex ||
               currentPresetIndex != lastPreset ||
               (int)isTraining != lastTraining ||
               (int)predictionMode != lastPredictionMode ||
               (int)currentNetworkMode != lastNetworkMode ||
               (int)CurrentNetworkHasSavedMapping() != lastSaved ||
               displayB0 != lastB0 ||
               displayB1 != lastB1 ||
               displayB2 != lastB2 ||
               displayB3 != lastB3){

                needRedraw = true;

                lastMenuIndex = menuIndex;
                lastPreset = currentPresetIndex;
                lastTraining = (int)isTraining;
                lastPredictionMode = (int)predictionMode;
                lastNetworkMode = (int)currentNetworkMode;
                lastSaved = (int)CurrentNetworkHasSavedMapping();
                lastB0 = displayB0;
                lastB1 = displayB1;
                lastB2 = displayB2;
                lastB3 = displayB3;
            }

            if(needRedraw){
                ScreenDrawMenu(
                    menuIndex,
                    tonePresets[currentPresetIndex].toneName,
                    currentNetworkMode == NETWORK_NEURALNET ? "NeuralNet" : "MiniCAB",
                    (int)isTraining,
                    (int)predictionMode,
                    (int)CurrentNetworkHasSavedMapping(),
                    displayB0,
                    displayB1,
                    displayB2,
                    displayB3
                );
            }
        }

        uint32_t now = System::GetNow();

        if(now - lastPrintTime >= 1000) {
            lastPrintTime = now;

            hw.seed.PrintLine("Network:%s Label:%d Preset:%s Phrase:%d Frames:%d Pred:%d Stable:%d Hold:%d Predict:%d Saved:%d",
                currentNetworkMode == NETWORK_NEURALNET ? "NeuralNet" : "MiniCAB",
                currentLabel,
                tonePresets[currentPresetIndex].toneName,
                (int)phraseActive,
                phraseFrames,
                nnOutput.predictedClass,
                predictedStableClass,
                predictedHoldCount,
                (int)predictionMode,
                (int)CurrentNetworkHasSavedMapping());

            hw.seed.PrintLine("Scores D:%d C:%d A:%d R:%d",
                (int)(nnOutput.scores[0] * 1000.0f),
                (int)(nnOutput.scores[1] * 1000.0f),
                (int)(nnOutput.scores[2] * 1000.0f),
                (int)(nnOutput.scores[3] * 1000.0f));
        }
    }
}