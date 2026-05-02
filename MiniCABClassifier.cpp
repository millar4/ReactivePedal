#include "MiniCABClassifier.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

static float ClipGrad(float x){
    if(x > 5.0f){
        return 5.0f;
    }

    if(x < -5.0f){
        return -5.0f;
    }

    return x;
}

void MiniCABClassifier::Init(){
    for(int i = 0; i < inputSize; i++){
        normMean[i] = 0.0f;
        normStd[i] = 1.0f;
    }

    hasNormalisation = false;

    for(int k = 0; k < numClasses; k++){
        classifierB[k] = 0.0f;

        for(int j = 0; j < representationSize; j++){
            classifierW[k][j] = 0.0f;
        }
    }

    Reset();
}

void MiniCABClassifier::Reset(){
    frameIndex = 0;
    frameCount = 0;
    std::memset(frameBuffer, 0, sizeof(frameBuffer));
}

void MiniCABClassifier::PushFrame(const AudioFeatures& f){
    frameBuffer[frameIndex] = f;
    frameIndex = (frameIndex + 1) % timeSteps;

    if(frameCount < timeSteps){
        frameCount++;
    }
}

CABOutput MiniCABClassifier::PredictSequence() const{
    CABOutput out = {};

    if(frameCount == 0){
        return out;
    }

    float seq[timeSteps][inputSize];
    float rep[representationSize];
    float probs[numClasses];

    BuildSequenceFromBuffer(seq);
    BuildRepresentation(seq, rep);
    ClassifierPredict(rep, probs);

    int bestClass = 0;
    float bestScore = probs[0];

    for(int i = 1; i < numClasses; i++){
        if(probs[i] > bestScore){
            bestScore = probs[i];
            bestClass = i;
        }
    }

    for(int i = 0; i < numClasses; i++){
        out.scores[i] = probs[i];
    }

    out.predictedClass = bestClass;
    return out;
}

void MiniCABClassifier::FeaturesToInput(const AudioFeatures& f, float input[inputSize]) const{
    input[0]  = f.rms;
    input[1]  = f.peak;
    input[2]  = f.zcr;
    input[3]  = f.spectralCentroid;
    input[4]  = f.spectralFlux;
    input[5]  = f.rmsDelta;
    input[6]  = f.envelope;
    input[7]  = f.envelopeDelta;
    input[8]  = f.rmsVariance;
    input[9]  = f.centroidVariance;
    input[10] = f.onsetCount;
    input[11] = f.timeSinceLastOnset;

    if(hasNormalisation){
        for(int i = 0; i < inputSize; i++){
            input[i] = (input[i] - normMean[i]) / normStd[i];
            input[i] = ClipValue(input[i], -8.0f, 8.0f);
        }
    }
}

void MiniCABClassifier::BuildSequenceFromBuffer(float seq[timeSteps][inputSize]) const{
    for(int t = 0; t < timeSteps; t++){
        for(int j = 0; j < inputSize; j++){
            seq[t][j] = 0.0f;
        }
    }

    int startSlot = timeSteps - frameCount;
    int oldestIndex = (frameIndex - frameCount + timeSteps) % timeSteps;

    for(int t = 0; t < frameCount; t++){
        int srcIndex = (oldestIndex + t) % timeSteps;
        int dstIndex = startSlot + t;

        FeaturesToInput(frameBuffer[srcIndex], seq[dstIndex]);
    }
}

void MiniCABClassifier::BuildSequenceWithLatest(const AudioFeatures& f, float seq[timeSteps][inputSize]) const{
    for(int t = 0; t < timeSteps; t++){
        for(int j = 0; j < inputSize; j++){
            seq[t][j] = 0.0f;
        }
    }

    int previousCount = frameCount;

    if(previousCount > timeSteps - 1){
        previousCount = timeSteps - 1;
    }

    int startSlot = timeSteps - previousCount - 1;
    int oldestIndex = (frameIndex - previousCount + timeSteps) % timeSteps;

    for(int t = 0; t < previousCount; t++){
        int srcIndex = (oldestIndex + t) % timeSteps;
        int dstIndex = startSlot + t;

        FeaturesToInput(frameBuffer[srcIndex], seq[dstIndex]);
    }

    FeaturesToInput(f, seq[timeSteps - 1]);
}

void MiniCABClassifier::BuildRepresentation(const float seq[timeSteps][inputSize], float rep[representationSize]) const{
    for(int i = 0; i < representationSize; i++){
        rep[i] = 0.0f;
    }

    for(int f = 0; f < numConvFilters; f++){
        float maxVal = -1000000000.0f;
        float sumVal = 0.0f;
        float lastVal = 0.0f;

        for(int t = 0; t < convTimeSteps; t++){
            float v = ConvFilterValue(seq, f, t);
            v = tanhf(v);

            if(v > maxVal){
                maxVal = v;
            }

            sumVal += v;
            lastVal = v;
        }

        rep[f] = sumVal / (float)convTimeSteps;
        rep[f + numConvFilters] = maxVal;
        rep[f + 2 * numConvFilters] = lastVal;
    }

    float mean[inputSize];
    float maxVal[inputSize];
    float minVal[inputSize];
    float motion[inputSize];

    for(int j = 0; j < inputSize; j++){
        mean[j] = 0.0f;
        maxVal[j] = -1000000000.0f;
        minVal[j] = 1000000000.0f;
        motion[j] = 0.0f;
    }

    float eventMax = -1000000000.0f;
    float gapMax = -1000000000.0f;

    for(int t = 0; t < timeSteps; t++){
        for(int j = 0; j < inputSize; j++){
            float v = seq[t][j];
            mean[j] += v;

            if(v > maxVal[j]){
                maxVal[j] = v;
            }

            if(v < minVal[j]){
                minVal[j] = v;
            }

            if(t > 0){
                motion[j] += fabsf(seq[t][j] - seq[t - 1][j]);
            }
        }

        float eventLogit = EventAttentionLogit(seq[t]);
        float gapLogit = GapAttentionLogit(seq[t]);

        if(eventLogit > eventMax){
            eventMax = eventLogit;
        }

        if(gapLogit > gapMax){
            gapMax = gapLogit;
        }
    }

    for(int j = 0; j < inputSize; j++){
        mean[j] /= (float)timeSteps;
        motion[j] /= (float)(timeSteps - 1);
    }

    float eventSum = 0.0f;
    float gapSum = 0.0f;
    float eventMean[inputSize];
    float gapMean[inputSize];

    for(int j = 0; j < inputSize; j++){
        eventMean[j] = 0.0f;
        gapMean[j] = 0.0f;
    }

    int activeFrames = 0;
    int quietFrames = 0;
    int longGapFrames = 0;

    for(int t = 0; t < timeSteps; t++){
        float eventWeight = expf(EventAttentionLogit(seq[t]) - eventMax);
        float gapWeight = expf(GapAttentionLogit(seq[t]) - gapMax);

        eventSum += eventWeight;
        gapSum += gapWeight;

        for(int j = 0; j < inputSize; j++){
            eventMean[j] += eventWeight * seq[t][j];
            gapMean[j] += gapWeight * seq[t][j];
        }

        if(seq[t][0] > 0.0f || seq[t][1] > 0.0f || seq[t][6] > 0.0f || seq[t][4] > 0.0f){
            activeFrames++;
        }

        if(seq[t][0] < 0.0f && seq[t][1] < 0.0f && seq[t][6] < 0.0f){
            quietFrames++;
        }

        if(seq[t][11] > 0.5f){
            longGapFrames++;
        }
    }

    if(eventSum < 1e-6f){
        eventSum = 1.0f;
    }

    if(gapSum < 1e-6f){
        gapSum = 1.0f;
    }

    for(int j = 0; j < inputSize; j++){
        eventMean[j] /= eventSum;
        gapMean[j] /= gapSum;
    }

    int base = numConvFilters * representationParts;

    rep[base + 0] = mean[0];
    rep[base + 1] = maxVal[0];
    rep[base + 2] = maxVal[0] - minVal[0];
    rep[base + 3] = motion[0];
    rep[base + 4] = mean[1];
    rep[base + 5] = maxVal[1];
    rep[base + 6] = mean[2];
    rep[base + 7] = mean[3];
    rep[base + 8] = maxVal[3] - minVal[3];
    rep[base + 9] = motion[3];
    rep[base + 10] = mean[4];
    rep[base + 11] = maxVal[4];
    rep[base + 12] = motion[4];
    rep[base + 13] = mean[6];
    rep[base + 14] = maxVal[6];
    rep[base + 15] = motion[6];
    rep[base + 16] = mean[7];
    rep[base + 17] = maxVal[7];
    rep[base + 18] = mean[5];
    rep[base + 19] = maxVal[5];
    rep[base + 20] = mean[10];
    rep[base + 21] = maxVal[10];
    rep[base + 22] = mean[11];
    rep[base + 23] = maxVal[11];
    rep[base + 24] = seq[timeSteps - 1][11];
    rep[base + 25] = maxVal[11] - minVal[11];
    rep[base + 26] = eventMean[0];
    rep[base + 27] = eventMean[1];
    rep[base + 28] = eventMean[4];
    rep[base + 29] = eventMean[10];
    rep[base + 30] = eventMean[11];
    rep[base + 31] = gapMean[0];
    rep[base + 32] = gapMean[1];
    rep[base + 33] = gapMean[4];
    rep[base + 34] = gapMean[10];
    rep[base + 35] = gapMean[11];
    rep[base + 36] = (float)activeFrames / (float)timeSteps;
    rep[base + 37] = (float)quietFrames / (float)timeSteps;
    rep[base + 38] = (float)longGapFrames / (float)timeSteps;
    rep[base + 39] = eventMean[11] - mean[11];

    for(int i = 0; i < representationSize; i++){
        rep[i] = ClipValue(rep[i], -8.0f, 8.0f);
    }
}

void MiniCABClassifier::ClassifierPredict(const float rep[representationSize], float probs[numClasses]) const{
    for(int k = 0; k < numClasses; k++){
        float sum = classifierB[k];

        for(int j = 0; j < representationSize; j++){
            sum += classifierW[k][j] * rep[j];
        }

        probs[k] = sum;
    }

    Softmax(probs, numClasses);
}

void MiniCABClassifier::TrainFromExample(const AudioFeatures& f, int label, float eta){
    float seq[timeSteps][inputSize];

    BuildSequenceWithLatest(f, seq);
    TrainOnSequence(seq, label, eta);
}

void MiniCABClassifier::TrainFromCurrentSequence(int label, float eta){
    if(frameCount == 0){
        return;
    }

    float seq[timeSteps][inputSize];

    BuildSequenceFromBuffer(seq);
    TrainOnSequence(seq, label, eta);
}

void MiniCABClassifier::TrainOnSequence(const float seq[timeSteps][inputSize], int label, float eta){
    if(label < 0 || label >= numClasses){
        return;
    }

    float rep[representationSize];
    float probs[numClasses];
    float target[numClasses];

    BuildRepresentation(seq, rep);
    ClassifierPredict(rep, probs);

    for(int k = 0; k < numClasses; k++){
        target[k] = 0.02f;
    }

    target[label] = 0.94f;

    float lr = eta;

    if(lr < 0.0001f){
        lr = 0.0001f;
    }

    if(lr > 0.05f){
        lr = 0.05f;
    }

    static constexpr float weightDecay = 0.00008f;

    for(int k = 0; k < numClasses; k++){
        float d = ClipGrad(probs[k] - target[k]);

        classifierB[k] -= lr * d;

        for(int j = 0; j < representationSize; j++){
            float grad = d * rep[j] + weightDecay * classifierW[k][j];
            classifierW[k][j] -= lr * ClipGrad(grad);
        }
    }
}

void MiniCABClassifier::SetNormalisation(const float mean[inputSize], const float std[inputSize]){
    for(int i = 0; i < inputSize; i++){
        normMean[i] = mean[i];
        normStd[i] = (std[i] < 1e-6f) ? 1.0f : std[i];
    }

    hasNormalisation = true;
}

float MiniCABClassifier::ClipValue(float x, float lo, float hi) const{
    if(x < lo){
        return lo;
    }

    if(x > hi){
        return hi;
    }

    return x;
}

float MiniCABClassifier::ConvFilterValue(const float seq[timeSteps][inputSize], int filterId, int t) const{
    float sum = 0.0f;

    for(int r = 0; r < convKernelSize; r++){
        int idx = t + r;

        switch(filterId){
            case 0:
                sum += seq[idx][0];
                break;
            case 1:
                sum += seq[idx][1];
                break;
            case 2:
                sum += seq[idx][4];
                break;
            case 3:
                sum += seq[idx][6];
                break;
            case 4:
                sum += 0.60f * seq[idx][10] + 0.40f * seq[idx][5];
                break;
            case 5:
                sum += seq[idx][11];
                break;
            case 6:
                sum += seq[idx][3];
                break;
            case 7:
                if(r == convKernelSize - 1){
                    sum += seq[idx][6] + 0.50f * seq[idx][4] + 0.30f * seq[idx][1];
                }
                else if(r == 0){
                    sum -= seq[idx][6] + 0.50f * seq[idx][4] + 0.30f * seq[idx][1];
                }
                break;
            default:
                break;
        }
    }

    if(filterId != 7){
        sum /= (float)convKernelSize;
    }

    return sum;
}

float MiniCABClassifier::EventAttentionLogit(const float input[inputSize]) const{
    float level = 0.35f * fabsf(input[0]) + 0.25f * fabsf(input[1]) + 0.25f * fabsf(input[6]);
    float movement = 0.70f * fabsf(input[4]) + 0.35f * fabsf(input[5]) + 0.35f * fabsf(input[7]) + 0.55f * fabsf(input[10]);
    float logit = -1.0f + level + movement;

    return ClipValue(logit, -8.0f, 8.0f);
}

float MiniCABClassifier::GapAttentionLogit(const float input[inputSize]) const{
    float quietness = -0.25f * fabsf(input[0]) - 0.20f * fabsf(input[1]) - 0.15f * fabsf(input[4]);
    float logit = 0.85f * input[11] + quietness;

    return ClipValue(logit, -8.0f, 8.0f);
}

void MiniCABClassifier::Softmax(float* values, int n) const{
    float maxVal = values[0];

    for(int i = 1; i < n; i++){
        if(values[i] > maxVal){
            maxVal = values[i];
        }
    }

    float sum = 0.0f;

    for(int i = 0; i < n; i++){
        values[i] = expf(values[i] - maxVal);
        sum += values[i];
    }

    if(sum > 0.0f){
        for(int i = 0; i < n; i++){
            values[i] /= sum;
        }
    }
}
