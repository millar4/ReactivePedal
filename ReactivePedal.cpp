#include "daisy_pod.h"
#include "daisysp.h"
#include "FeatureExtractor.h"
#include "NeuralNet.h"
#include <cmath>
#include <cstring>

using namespace daisy;
using namespace daisysp;
using namespace std;

DaisyPod hw;
FeatureExtractor extractor; //this is the feature extraction engine 
AudioFeatures latestFeatures; //raw 
AudioFeatures smoothedFeatures = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};


//Neural net, mapping creation
NeuralNet neuralnet;
NNOutput nnOutput = {};
MappingState savedMapping;

bool hasSavedMapping = false;
bool predictionMode = false;
bool isTraining = false;



int currentLabel = 0; //currently selected class
int predictedCandidate = 0; //best guess
int predictedStableClass = 0; //stable prediction
int predictedHoldCount = 0; //How many phrase evals in a row have predicted the same candidate
int stableClassAge = 0; // here we track how long the current stable class has lasted 
int candidateClassAge = 0; //how long has candidate class lasted

static constexpr int perClassBufferCapacity = 32; // up to 32 phrases 
AudioFeatures featureBuffer[4][perClassBufferCapacity]; // feature buffer , two dimensional array of four classes with up to 32 examples per class 
int bufferIndex[4] = {0, 0, 0, 0}; //write position for each class , creates a circular buffer
int bufferCount[4] = {0, 0, 0, 0}; //number of valid examples 



enum WhichEffect{
    MODE_DISTORTION = 0,
    MODE_CHORUS,
    MODE_AMBIENT,
    MODE_REVERB
};

//clip shapes 
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
static constexpr int bufferSize = 48000; //1s of audio
DSY_SDRAM_BSS float delayBuffer[bufferSize]; //for chorus, ambient and reverb 

int writeIndex = 0; //current index in circular delay buffer

float sampleRate = 48000.0f;
float baseDelayMs = 12.0f;
float depthMs = 4.0f;
float rateHz = 0.6f;
float chorusMix = 0.4f; //40% delay , 40% dry mix 

float ambientLevel = 7.0f;
float reverbLevel = 6.0f;


//millisecond -> sample count 
float baseDelay = sampleRate * baseDelayMs / 1000.0f;
float depth = sampleRate * depthMs / 1000.0f;


//sine wave modulation 
float phase = 0.0f; //curr angle 
float phaseIncrement = 2.0f * 3.14159265359f * rateHz / sampleRate; //each sample advances the LFO a tiny bit


//smoothed tone controls, code gradually moves towards these targets.
volatile float smoothedDrive = 1.0f;
volatile float smoothedTone = 0.3f;
volatile float smoothedPreamp = 3.0f;
volatile float smoothedMix = 0.9f;
volatile float smoothedLevel = 0.8f;
volatile float smoothedClipThreshold = 1.0f;


//use real signal
bool useGeneratedSignal = false;
float signalPhase = 0.0f;
float signalTime = 0.0f;


//our smoothed values move towards the target pre set values
float targetDrive = 1.0f;
float targetTone = 0.25f;
float targetPreamp = 3.0f;
float targetMix = 0.9f;
float targetLevel = 1.5f;
float targetClipThreshold = 1.0f;


//ambient params 
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


//reverb params
float reverbMix = 0.82f;
float reverbFeedback = 0.84f;
float reverbToneState = 0.0f;

//thus far we are using a three delay reverb , as an echo approximation

float reverbDelay1Ms = 140.0f;
float reverbDelay2Ms = 260.0f;
float reverbDelay3Ms = 420.0f;

//Convert to samples 
float reverbDelay1 = sampleRate * reverbDelay1Ms / 1000.0f;
float reverbDelay2 = sampleRate * reverbDelay2Ms / 1000.0f;
float reverbDelay3 = sampleRate * reverbDelay3Ms / 1000.0f;

ClipMode currentClipMode = CLIP_SOFT;
float toneStateL = 0.0f;


//An array of tone pre sets
static const TonePreset tonePresets[] = {
    {"Distortion", 4.2f, 4.8f, 0.18f, 0.97f, 1.25f, 0.75f, CLIP_ASYM},
    {"Chorus", 1.2f, 1.0f, 0.32f, 0.45f, 1.2f, 1.0f, CLIP_BYPASS},
    {"Ambient", 1.1f, 0.9f, 0.20f, 0.55f, 1.0f, 1.0f, CLIP_SOFT},
    {"Reverb", 1.4f, 1.1f, 0.24f, 0.65f, 1.1f, 1.0f, CLIP_SOFT},
};

static constexpr int noNumTonePresets = sizeof(tonePresets) / sizeof(tonePresets[0]);
int currentPresetIndex = 0;



float GenerateSignal();
float ambientProcessor(float x);
float reverbProcessor(float x);

//A phrase is a collection of frames treated as valid musical data, we group frames into phrases 
bool phraseActive = false; //is our phrase active 
uint32_t phraseStartMs = 0; //when did we begin
uint32_t lastActiveMs = 0;//the last time we considered our signal active
AudioFeatures phraseAccum = {}; //feature sums over phrase
int phraseFrames = 0; //how many feature frames do we have in this phrase
static constexpr uint32_t phraseEndGapMs = 160; //end of phrase (if inactive in ms)
static constexpr uint32_t maxPhraseLengthMs = 1200; //max phrase length
static constexpr int minPhraseFrames = 1; //each phrase needs at least one frame
static constexpr uint32_t phraseAttackCooldownMs = 180; //prevents attack detection from counting too many attacks together


//for debugging
AudioFeatures lastPhraseFeatures = {};
bool lastPhraseReady = false;


//phrase statistics to detect, arpeggiation, sustained ambient notes , etc. 
float phraseMaxRms = 0.0f; //Range through phrase
float phraseMinRms = 1.0f;
float phraseMaxCentroid = 0.0f; //Brightness change
float phraseMinCentroid = 1000000.0f;
float phraseMaxPeak = 0.0f;
int phraseAttackCount = 0; //how many note attacks
uint32_t lastPhraseAttackMs = 0;
uint32_t firstPhraseAttackMs = 0;
uint32_t prevPhraseAttackMs = 0;
float phraseAttackGapSumMs = 0.0f; //important for detecting sustained one note playing styles
int phraseAttackGapCount = 0;

static constexpr int targetPhrasesPerClass = 32; //ideal number of phrases per class
uint32_t lastAcceptedPhraseMs[4] = {0, 0, 0, 0};

//Classification arrays
float classDistances[4] = {9999.0f, 9999.0f, 9999.0f, 9999.0f}; //difference in current phrase vs class centroid (lower is better)
float classScores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float smoothedClassScores[4] = {0.25f, 0.25f, 0.25f, 0.25f};

float datasetMean[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float datasetStd[12] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};


//clamp function for safe audio clipping
float Clamp(float x, float lo, float hi){
    if(x < lo){
        return lo;
    }
    if(x > hi){
        return hi;
    }
    return x;
}

//inbound raw, smoothed out 
void SmoothFeatures(const AudioFeatures& in, AudioFeatures& out){
    out.rms                = 0.35f * out.rms + 0.65f * in.rms; //exponential smoothing 
    out.peak               = 0.35f * out.peak + 0.65f * in.peak;
    out.zcr                = 0.45f * out.zcr + 0.55f * in.zcr;
    out.spectralCentroid   = 0.45f * out.spectralCentroid + 0.55f * in.spectralCentroid;
    out.spectralFlux       = 0.30f * out.spectralFlux + 0.70f * in.spectralFlux;
    out.rmsDelta           = 0.25f * out.rmsDelta + 0.75f * in.rmsDelta;
    out.envelope           = 0.35f * out.envelope + 0.65f * in.envelope;
    out.envelopeDelta      = 0.25f * out.envelopeDelta + 0.75f * in.envelopeDelta;
    out.rmsVariance        = 0.45f * out.rmsVariance + 0.55f * in.rmsVariance;
    out.centroidVariance   = 0.45f * out.centroidVariance + 0.55f * in.centroidVariance;
    out.onsetCount         = 0.15f * out.onsetCount + 0.85f * in.onsetCount; //need heavy weights towards last value
    out.timeSinceLastOnset = 0.15f * out.timeSinceLastOnset + 0.85f * in.timeSinceLastOnset;
}


//used to restore all phrase values back to default, training starts/ stops or prediction starts
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


//called for each frame during a phrase
void AccumulatePhrase(const AudioFeatures& f, uint32_t nowMs){

    //f is one frame in this function
    //Below is the accumulation section
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

    //extremity tracking
    //track highest RMS seen , lowest RMS seen, highest centroid, lowest centroid etc, this allows us to see how dynamic the phrase was
    if(f.rms > phraseMaxRms) phraseMaxRms = f.rms;
    if(f.rms < phraseMinRms) phraseMinRms = f.rms;
    if(f.spectralCentroid > phraseMaxCentroid) phraseMaxCentroid = f.spectralCentroid;
    if(f.spectralCentroid < phraseMinCentroid) phraseMinCentroid = f.spectralCentroid;
    if(f.peak > phraseMaxPeak) phraseMaxPeak = f.peak;

    //do we have a strong attack, strong amplitude, significant frequency spectrum changes from previous frame
    //alternatively we could have have strongly rising amplitude envelope or changes in harmonic content
    //overall this detects the start of a plucked note 
    bool strongAttack = f.peak > 0.0075f && (f.spectralFlux > 0.035f || f.envelopeDelta > 0.012f || f.rmsDelta > 0.010f);

    bool cooldownExpired =
        lastPhraseAttackMs == 0 ||
        (nowMs - lastPhraseAttackMs) > phraseAttackCooldownMs;

    if(strongAttack && cooldownExpired){
        if(phraseAttackCount == 0){
            firstPhraseAttackMs = nowMs;
        }
        else{
            //detect distance between attacks
            phraseAttackGapSumMs += (float)(nowMs - prevPhraseAttackMs);
            phraseAttackGapCount++;
        }
        
        phraseAttackCount++;
        prevPhraseAttackMs = nowMs;
        lastPhraseAttackMs = nowMs;
    }
}

//create audio features summary
void FinalisePhrase(AudioFeatures& out){
    //zero out the output 
    std::memset(&out, 0, sizeof(AudioFeatures));
    if(phraseFrames <= 0){
        return;
    }

    float inv = 1.0f / (float)phraseFrames;

    //average frame based values, not everything is the average though

    out.rms = phraseAccum.rms * inv; //mean RMS
    out.peak = phraseMaxPeak; //use maximum peak of the phrase
    out.zcr = phraseAccum.zcr * inv;
    out.spectralCentroid = phraseAccum.spectralCentroid * inv; //
    out.spectralFlux = phraseAccum.spectralFlux * inv;
    out.envelope = phraseAccum.envelope * inv;
    out.rmsVariance = phraseMaxRms - phraseMinRms; //phrase change over time
    out.centroidVariance = phraseMaxCentroid - phraseMinCentroid; //phrase change over time


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
    //how much of phrase was 'tail' after last attack
    float sustainRatio = tailGapSec / phraseDurationSec;

    if(sustainRatio > 1.0f){
        sustainRatio = 1.0f;
    }
    out.onsetCount = (float)phraseAttackCount; //num of attacks 
    //repurposed to mean attacksPerSecond
    out.rmsDelta = attacksPerSecond;
    //repurpose to sustain detector
    out.envelopeDelta = sustainRatio;
    out.timeSinceLastOnset = tailGapSec; //tail duration
}


//is the current frame active enough
bool IsPhraseSignalActive(const AudioFeatures& f){
    return
        f.rms > 0.0030f ||
        f.peak > 0.0060f ||
        f.envelope > 0.0030f;
}

//functoin rejecting weak training labels
bool ShouldAcceptTrainingPhrase(int label, const AudioFeatures& phraseFeatures, uint32_t nowMs){
    (void)nowMs;

    //self explainatory
    if(bufferCount[label] >= targetPhrasesPerClass){
        hw.seed.PrintLine("Reject reason: class full");
        return false;
    }

    if(label == MODE_DISTORTION){
        bool ok =
            phraseFeatures.rms > 0.0040f &&
            phraseFeatures.peak > 0.0080f &&
            phraseFeatures.spectralFlux > 0.040f;

        if(!ok){
            hw.seed.PrintLine("Reject reason: weak distortion RMS:%d Peak:%d Flux:%d",
                (int)(phraseFeatures.rms * 1000.0f),
                (int)(phraseFeatures.peak * 1000.0f),
                (int)(phraseFeatures.spectralFlux * 1000.0f));
        }
        return ok;
    }
    else if(label == MODE_CHORUS){
        bool ok =
            phraseFeatures.rms > 0.0015f &&
            phraseFeatures.peak > 0.0035f &&
            phraseFeatures.spectralFlux > 0.012f &&
            phraseFeatures.onsetCount >= 2.0f &&
            phraseFeatures.rmsDelta > 2.0f &&
            phraseFeatures.envelopeDelta < 0.45f;

        if(!ok){
            hw.seed.PrintLine("Reject reason: weak chorus RMS:%d Peak:%d Flux:%d On:%d APS:%d Sus:%d Tail:%d",
                (int)(phraseFeatures.rms * 1000.0f),
                (int)(phraseFeatures.peak * 1000.0f),
                (int)(phraseFeatures.spectralFlux * 1000.0f),
                (int)(phraseFeatures.onsetCount * 1000.0f),
                (int)(phraseFeatures.rmsDelta * 1000.0f),
                (int)(phraseFeatures.envelopeDelta * 1000.0f),
                (int)(phraseFeatures.timeSinceLastOnset * 1000.0f));
        }
        return ok;
    }
    else if(label == MODE_AMBIENT){
        bool ok =
            phraseFeatures.rms > 0.0004f &&
            phraseFeatures.peak > 0.0010f &&
            phraseFeatures.onsetCount <= 2.5f &&
            phraseFeatures.envelopeDelta > 0.12f;

        if(!ok){
            hw.seed.PrintLine("Reject reason: weak ambient RMS:%d Peak:%d On:%d APS:%d Sus:%d Tail:%d",
                (int)(phraseFeatures.rms * 1000.0f),
                (int)(phraseFeatures.peak * 1000.0f),
                (int)(phraseFeatures.onsetCount * 1000.0f),
                (int)(phraseFeatures.rmsDelta * 1000.0f),
                (int)(phraseFeatures.envelopeDelta * 1000.0f),
                (int)(phraseFeatures.timeSinceLastOnset * 1000.0f));
        }
        return ok;
    }
    else if(label == MODE_REVERB){
        bool ok =
            phraseFeatures.rms > 0.0025f &&
            phraseFeatures.peak > 0.0050f;

        if(!ok){
            hw.seed.PrintLine("Reject reason: weak reverb RMS:%d Peak:%d",
                (int)(phraseFeatures.rms * 1000.0f),
                (int)(phraseFeatures.peak * 1000.0f));
        }
        return ok;
    }

    return false;
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

bool ComputeDatasetNormalisation(){
    for(int i = 0; i < 12; i++){
        datasetMean[i] = 0.0f;
        datasetStd[i] = 1.0f;
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

int CountReadyClasses(){
    int readyClasses = 0;
    for(int i = 0; i < 4; i++){
        if(bufferCount[i] >= 1){
            readyClasses++;
        }
    }
    return readyClasses;
}

bool TrainNetworkFromBuffers(){
    if(CountReadyClasses() < 2){
        return false;
    }

    if(!ComputeDatasetNormalisation()){
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

//main classification loop
void ClassifyByNeuralNet(const AudioFeatures& phraseFeatures){
    nnOutput = neuralnet.Predict(phraseFeatures);

    for(int i = 0; i < 4; i++){
        classScores[i] = nnOutput.scores[i];
        classDistances[i] = 1.0f - nnOutput.scores[i];
    }
}

//Poor forms of classification
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
    }

    nnOutput.predictedClass = currentPresetIndex;
}

void UpdatePredictionState(bool allowImmediateSwitch, const AudioFeatures& phraseFeatures){
    if(IsTransitionOnlyPhrase(phraseFeatures)){
        return;
    }

    int rawBestClass = 0;
    float rawBestScore = nnOutput.scores[0];
    float rawSecondScore = 0.0f;

    for(int i = 1; i < 4; i++){
        float s = nnOutput.scores[i];
        if(s > rawBestScore){
            rawSecondScore = rawBestScore;
            rawBestScore = s;
            rawBestClass = i;
        }
        else if(s > rawSecondScore){
            rawSecondScore = s;
        }
    }

    float rawMargin = rawBestScore - rawSecondScore;

    for(int i = 0; i < 4; i++){
        smoothedClassScores[i] = 0.45f * smoothedClassScores[i] + 0.55f * nnOutput.scores[i];
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

    float margin = bestScore - secondScore;

    bool confident =
        rawBestScore > 0.45f &&
        rawMargin > 0.08f &&
        bestScore > 0.34f &&
        margin > 0.05f;

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

    bool strongAmbientStyle =
        predictedCandidate == MODE_AMBIENT &&
        phraseFeatures.onsetCount <= 1.5f &&
        phraseFeatures.envelopeDelta > 0.18f &&
        phraseFeatures.timeSinceLastOnset > 0.10f;

    int requiredHold = 2;

    if(allowImmediateSwitch && rawBestScore > 0.72f && rawMargin > 0.18f){
        requiredHold = 1;
    }

    if(predictedStableClass != predictedCandidate && bestScore < 0.40f){
        requiredHold = 3;
    }

    if(strongAmbientStyle){
        requiredHold = 1;
    }

    if(predictedHoldCount >= requiredHold){
        if(predictedStableClass != predictedCandidate){
            predictedStableClass = predictedCandidate;
            stableClassAge = 0;

            for(int i = 0; i < 4; i++){
                smoothedClassScores[i] = 0.05f;
            }
            smoothedClassScores[predictedStableClass] = bestScore;
        }
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

float ProcessTone(float x){
    float pre = x * smoothedPreamp;
    float drive = pre * smoothedDrive;
    float wet = ApplyClipper(drive, currentClipMode, smoothedClipThreshold);
    float mixed = (1.0f - smoothedMix) * x + smoothedMix * wet;
    toneStateL = toneStateL + smoothedTone * (mixed - toneStateL);
    float y = toneStateL * smoothedLevel;
    return y;
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
            y = ((1.0f - chorusMix) * x + chorusMix * delayed) * 1.5f;
            delayBuffer[writeIndex] = x;
            writeIndex = (writeIndex + 1) % bufferSize;
            phase += phaseIncrement;
            if(phase >= 6.28318530718f) phase -= 6.28318530718f;
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
    if(ambientPhase >= 6.28318530718f) ambientPhase -= 6.28318530718f;

    float y = ((1.0f - ambientMix) * x + ambientMix * softened) * ambientLevel;
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

    float y = ((1.0f - reverbMix) * x + reverbMix * echoes) * reverbLevel;
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

    if(signalPhase >= 6.28318530718f) signalPhase -= 6.28318530718f;

    signalTime += 1.0f / sampleRate;

    if(currEffectMode == MODE_AMBIENT){
        if(signalTime >= 5.0f) signalTime = 0.0f;
    }
    else if(currEffectMode == MODE_REVERB){
        if(signalTime >= 3.0f) signalTime = 0.0f;
    }
    else{
        if(signalTime >= 1.5f) signalTime = 0.0f;
    }

    return x;
}

int main(void){
    hw.Init();
    hw.seed.StartLog();
    System::Delay(3000);

    hw.SetAudioBlockSize(48);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    extractor.Init(256, 48000.0f);
    neuralnet.Init();

    ApplyPresetIndex(currentPresetIndex);
    ResetPhrase();
    ResetPredictionState();

    hw.seed.PrintLine("Starting...");
    hw.seed.PrintLine("Preset: %s", tonePresets[currentPresetIndex].toneName);
    hw.StartAudio(AudioCallBack);
    hw.seed.PrintLine("Audio started");

    uint32_t lastPrintTime = 0;
    int modeIndex = 0;

    while(1) {
        hw.ProcessAnalogControls();
        hw.ProcessDigitalControls();

        if(hw.button1.RisingEdge()){
            isTraining = !isTraining;

            if(isTraining){
                predictionMode = false;
                ResetPhrase();
                ResetPredictionState();
                hw.seed.PrintLine("Training ON Label:%d Preset:%s",
                    currentLabel,
                    tonePresets[currentLabel].toneName);
            }
            else{
                ResetPhrase();
                hw.seed.PrintLine("Training OFF");
            }
        }

        if(hw.button2.RisingEdge()){
            int readyClasses = CountReadyClasses();

            if(readyClasses < 2){
                hw.seed.PrintLine("Need 2 trained classes before saving");
            }
            else{
                bool trainedOk = TrainNetworkFromBuffers();

                if(trainedOk){
                    hasSavedMapping = true;
                    predictionMode = true;
                    isTraining = false;
                    neuralnet.LoadState(savedMapping);
                    ResetPhrase();
                    ResetPredictionState();
                    hw.seed.PrintLine("Saved mapping and entered prediction mode");
                }
                else{
                    hw.seed.PrintLine("Training failed");
                }
            }
        }

        int inc = hw.encoder.Increment();
        if(inc > 0){
            modeIndex = (modeIndex + 1) % 4;
        }
        else if(inc < 0){
            modeIndex = (modeIndex + 3) % 4;
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

            uint32_t nowMs = System::GetNow();
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

                if(predictionMode && hasSavedMapping && phraseFrames >= 2){
                    AudioFeatures livePhraseFeatures;
                    FinalisePhrase(livePhraseFeatures);
                    ClassifyByNeuralNet(livePhraseFeatures);
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
                    if(isTraining){
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
                    else if(predictionMode && hasSavedMapping){
                        ClassifyByNeuralNet(phraseFeatures);
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

            smoothedDrive = 0.70f * smoothedDrive + 0.30f * targetDrive;
            smoothedTone = 0.70f * smoothedTone + 0.30f * targetTone;
            smoothedPreamp = 0.70f * smoothedPreamp + 0.30f * targetPreamp;
            smoothedMix = 0.70f * smoothedMix + 0.30f * targetMix;
            smoothedLevel = 0.70f * smoothedLevel + 0.30f * targetLevel;
            smoothedClipThreshold = 0.70f * smoothedClipThreshold + 0.30f * targetClipThreshold;
        }

        if(isTraining){
            ApplyPresetIndex(currentLabel);
        }
        else if(predictionMode && hasSavedMapping){
            ApplyPresetIndex(predictedStableClass);
        }
        else{
            ApplyPresetIndex(currentLabel);
        }

        uint32_t now = System::GetNow();
        if(now - lastPrintTime >= 1000) {
            lastPrintTime = now;

            hw.seed.PrintLine("Label:%d Preset:%s Phrase:%d Frames:%d Pred:%d Stable:%d Hold:%d",
                currentLabel,
                tonePresets[currentPresetIndex].toneName,
                (int)phraseActive,
                phraseFrames,
                nnOutput.predictedClass,
                predictedStableClass,
                predictedHoldCount);

            hw.seed.PrintLine("RMS:%d Peak:%d Flux:%d On:%d Gap:%d",
                (int)(smoothedFeatures.rms * 1000.0f),
                (int)(smoothedFeatures.peak * 1000.0f),
                (int)(smoothedFeatures.spectralFlux * 1000.0f),
                (int)(smoothedFeatures.onsetCount * 1000.0f),
                (int)(smoothedFeatures.timeSinceLastOnset * 1000.0f));

            hw.seed.PrintLine("LastPhrase RMS:%d Peak:%d Flux:%d On:%d APS:%d Sus:%d Tail:%d",
                (int)(lastPhraseFeatures.rms * 1000.0f),
                (int)(lastPhraseFeatures.peak * 1000.0f),
                (int)(lastPhraseFeatures.spectralFlux * 1000.0f),
                (int)(lastPhraseFeatures.onsetCount * 1000.0f),
                (int)(lastPhraseFeatures.rmsDelta * 1000.0f),
                (int)(lastPhraseFeatures.envelopeDelta * 1000.0f),
                (int)(lastPhraseFeatures.timeSinceLastOnset * 1000.0f));

            hw.seed.PrintLine("Dist:%d Chor:%d Amb:%d Rev:%d | D:%d C:%d A:%d R:%d | Train:%d B0:%d B1:%d B2:%d B3:%d Saved:%d Predict:%d",
                (int)(classDistances[0] * 100.0f),
                (int)(classDistances[1] * 100.0f),
                (int)(classDistances[2] * 100.0f),
                (int)(classDistances[3] * 100.0f),
                (int)(classScores[0] * 100.0f),
                (int)(classScores[1] * 100.0f),
                (int)(classScores[2] * 100.0f),
                (int)(classScores[3] * 100.0f),
                (int)isTraining,
                bufferCount[0],
                bufferCount[1],
                bufferCount[2],
                bufferCount[3],
                (int)hasSavedMapping,
                (int)predictionMode);
        }
    }
}