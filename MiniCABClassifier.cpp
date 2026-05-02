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

    for(int f = 0; f < numConvFilters; f++){
        convB[f] = 0.0f;

        for(int t = 0; t < convKernelSize; t++){
            for(int j = 0; j < inputSize; j++){
                convW[f][t][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            }
        }
    }

    for(int cl = 0; cl < numClassifiers; cl++){
        for(int i = 0; i < classifierHiddenSize; i++){
            classifierB1[cl][i] = 0.0f;

            for(int j = 0; j < representationSize; j++){
                classifierW1[cl][i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            }
        }

        for(int i = 0; i < numClasses; i++){
            classifierB2[cl][i] = 0.0f;

            for(int j = 0; j < classifierHiddenSize; j++){
                classifierW2[cl][i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            }
        }
    }

    for(int i = 0; i < attentionHiddenSize; i++){
        attentionB1[i] = 0.0f;

        for(int j = 0; j < representationSize; j++){
            attentionW1[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        }
    }

    for(int i = 0; i < numClassifiers; i++){
        attentionB2[i] = 0.0f;

        for(int j = 0; j < attentionHiddenSize; j++){
            attentionW2[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
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
    float convPre[numConvFilters][convTimeSteps];
    float convAct[numConvFilters][convTimeSteps];
    float rep[representationSize];
    int maxIndex[numConvFilters];

    BuildSequenceFromBuffer(seq);
    CNNForward(seq, convPre, convAct, rep, maxIndex);

    float alpha[numClassifiers];
    AttentionUnitPredict(rep, alpha);

    float finalScores[numClasses] = {0.0f};

    for(int cl = 0; cl < numClassifiers; cl++){
        float probs[numClasses];
        ClassifierPredict(cl, rep, probs);

        for(int k = 0; k < numClasses; k++){
            finalScores[k] += alpha[cl] * probs[k];
        }
    }

    int bestClass = 0;
    float bestScore = finalScores[0];

    for(int i = 1; i < numClasses; i++){
        if(finalScores[i] > bestScore){
            bestScore = finalScores[i];
            bestClass = i;
        }
    }

    for(int i = 0; i < numClasses; i++){
        out.scores[i] = finalScores[i];
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

            if(input[i] > 8.0f){
                input[i] = 8.0f;
            }

            if(input[i] < -8.0f){
                input[i] = -8.0f;
            }
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

void MiniCABClassifier::CNNForward(const float seq[timeSteps][inputSize],
                                   float convPre[numConvFilters][convTimeSteps],
                                   float convAct[numConvFilters][convTimeSteps],
                                   float rep[representationSize],
                                   int maxIndex[numConvFilters]) const{
    for(int i = 0; i < representationSize; i++){
        rep[i] = 0.0f;
    }

    for(int f = 0; f < numConvFilters; f++){
        float maxVal = -1000000000.0f;
        float sumAct = 0.0f;
        int bestIndex = 0;

        for(int t = 0; t < convTimeSteps; t++){
            float sum = convB[f];

            for(int r = 0; r < convKernelSize; r++){
                for(int j = 0; j < inputSize; j++){
                    sum += convW[f][r][j] * seq[t + r][j];
                }
            }

            convPre[f][t] = sum;
            convAct[f][t] = ReLU(sum);
            sumAct += convAct[f][t];

            if(convAct[f][t] > maxVal){
                maxVal = convAct[f][t];
                bestIndex = t;
            }
        }

        float meanVal = sumAct / (float)convTimeSteps;
        float firstVal = convAct[f][0];
        float lastVal = convAct[f][convTimeSteps - 1];
        float changeVal = lastVal - firstVal;

        rep[f] = maxVal;
        rep[f + numConvFilters] = meanVal;
        rep[f + 2 * numConvFilters] = lastVal;
        rep[f + 3 * numConvFilters] = changeVal;

        maxIndex[f] = bestIndex;
    }

    int base = numConvFilters * representationParts;

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
    }

    for(int j = 0; j < inputSize; j++){
        mean[j] /= (float)timeSteps;
        motion[j] /= (float)(timeSteps - 1);
    }

    rep[base + 0] = mean[0];
    rep[base + 1] = maxVal[0];
    rep[base + 2] = maxVal[0] - minVal[0];
    rep[base + 3] = mean[1];
    rep[base + 4] = maxVal[1] - minVal[1];
    rep[base + 5] = mean[2];
    rep[base + 6] = mean[3];
    rep[base + 7] = motion[3];
    rep[base + 8] = maxVal[3] - minVal[3];
    rep[base + 9] = mean[4];
    rep[base + 10] = maxVal[4];
    rep[base + 11] = motion[4];
    rep[base + 12] = maxVal[4] - minVal[4];
    rep[base + 13] = motion[6] + maxVal[6] - minVal[6];
    rep[base + 14] = mean[10];
    rep[base + 15] = mean[11];
}

void MiniCABClassifier::ClassifierPredict(int classifierId, const float rep[representationSize], float probs[numClasses]) const{
    float hidden[classifierHiddenSize];

    for(int i = 0; i < classifierHiddenSize; i++){
        float sum = classifierB1[classifierId][i];

        for(int j = 0; j < representationSize; j++){
            sum += classifierW1[classifierId][i][j] * rep[j];
        }

        hidden[i] = ReLU(sum);
    }

    for(int i = 0; i < numClasses; i++){
        float sum = classifierB2[classifierId][i];

        for(int j = 0; j < classifierHiddenSize; j++){
            sum += classifierW2[classifierId][i][j] * hidden[j];
        }

        probs[i] = sum;
    }

    Softmax(probs, numClasses);
}

void MiniCABClassifier::AttentionUnitPredict(const float rep[representationSize], float alpha[numClassifiers]) const{
    float hidden[attentionHiddenSize];

    for(int i = 0; i < attentionHiddenSize; i++){
        float sum = attentionB1[i];

        for(int j = 0; j < representationSize; j++){
            sum += attentionW1[i][j] * rep[j];
        }

        hidden[i] = ReLU(sum);
    }

    for(int i = 0; i < numClassifiers; i++){
        float sum = attentionB2[i];

        for(int j = 0; j < attentionHiddenSize; j++){
            sum += attentionW2[i][j] * hidden[j];
        }

        alpha[i] = sum;
    }

    Softmax(alpha, numClassifiers);
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

    float convPre[numConvFilters][convTimeSteps];
    float convAct[numConvFilters][convTimeSteps];
    float rep[representationSize];
    int maxIndex[numConvFilters];

    CNNForward(seq, convPre, convAct, rep, maxIndex);

    float classifierHiddenPre[numClassifiers][classifierHiddenSize];
    float classifierHidden[numClassifiers][classifierHiddenSize];
    float classifierProbs[numClassifiers][numClasses];

    for(int cl = 0; cl < numClassifiers; cl++){
        for(int i = 0; i < classifierHiddenSize; i++){
            float sum = classifierB1[cl][i];

            for(int j = 0; j < representationSize; j++){
                sum += classifierW1[cl][i][j] * rep[j];
            }

            classifierHiddenPre[cl][i] = sum;
            classifierHidden[cl][i] = ReLU(sum);
        }

        for(int k = 0; k < numClasses; k++){
            float sum = classifierB2[cl][k];

            for(int j = 0; j < classifierHiddenSize; j++){
                sum += classifierW2[cl][k][j] * classifierHidden[cl][j];
            }

            classifierProbs[cl][k] = sum;
        }

        Softmax(classifierProbs[cl], numClasses);
    }

    float attentionHiddenPre[attentionHiddenSize];
    float attentionHidden[attentionHiddenSize];

    for(int i = 0; i < attentionHiddenSize; i++){
        float sum = attentionB1[i];

        for(int j = 0; j < representationSize; j++){
            sum += attentionW1[i][j] * rep[j];
        }

        attentionHiddenPre[i] = sum;
        attentionHidden[i] = ReLU(sum);
    }

    float alpha[numClassifiers];

    for(int cl = 0; cl < numClassifiers; cl++){
        float sum = attentionB2[cl];

        for(int j = 0; j < attentionHiddenSize; j++){
            sum += attentionW2[cl][j] * attentionHidden[j];
        }

        alpha[cl] = sum;
    }

    Softmax(alpha, numClassifiers);

    float finalScores[numClasses] = {0.0f};

    for(int k = 0; k < numClasses; k++){
        for(int cl = 0; cl < numClassifiers; cl++){
            finalScores[k] += alpha[cl] * classifierProbs[cl][k];
        }
    }

    float target[numClasses];

    for(int k = 0; k < numClasses; k++){
        target[k] = 0.02f;
    }

    target[label] = 0.94f;

    float dFinal[numClasses];

    for(int k = 0; k < numClasses; k++){
        dFinal[k] = finalScores[k] - target[k];
    }

    float dClassifierProbs[numClassifiers][numClasses];

    for(int cl = 0; cl < numClassifiers; cl++){
        for(int k = 0; k < numClasses; k++){
            dClassifierProbs[cl][k] = alpha[cl] * dFinal[k];
        }
    }

    float dAlpha[numClassifiers] = {0.0f};

    for(int cl = 0; cl < numClassifiers; cl++){
        for(int k = 0; k < numClasses; k++){
            dAlpha[cl] += dFinal[k] * classifierProbs[cl][k];
        }
    }

    float dAttentionLogits[numClassifiers];
    float alphaDotGrad = 0.0f;

    for(int cl = 0; cl < numClassifiers; cl++){
        alphaDotGrad += dAlpha[cl] * alpha[cl];
    }

    for(int cl = 0; cl < numClassifiers; cl++){
        dAttentionLogits[cl] = alpha[cl] * (dAlpha[cl] - alphaDotGrad);
        dAttentionLogits[cl] = ClipGrad(dAttentionLogits[cl]);
    }

    float dRep[representationSize] = {0.0f};

    float dAttentionHidden[attentionHiddenSize] = {0.0f};

    for(int cl = 0; cl < numClassifiers; cl++){
        for(int j = 0; j < attentionHiddenSize; j++){
            dAttentionHidden[j] += dAttentionLogits[cl] * attentionW2[cl][j];
        }
    }

    float dAttentionPre[attentionHiddenSize];

    for(int j = 0; j < attentionHiddenSize; j++){
        dAttentionPre[j] = dAttentionHidden[j] * ReLUGrad(attentionHiddenPre[j]);
        dAttentionPre[j] = ClipGrad(dAttentionPre[j]);

        for(int m = 0; m < representationSize; m++){
            dRep[m] += dAttentionPre[j] * attentionW1[j][m];
        }
    }

    for(int cl = 0; cl < numClassifiers; cl++){
        float dClassifierLogits[numClasses];
        float probDotGrad = 0.0f;

        for(int k = 0; k < numClasses; k++){
            probDotGrad += dClassifierProbs[cl][k] * classifierProbs[cl][k];
        }

        for(int k = 0; k < numClasses; k++){
            dClassifierLogits[k] = classifierProbs[cl][k] * (dClassifierProbs[cl][k] - probDotGrad);
            dClassifierLogits[k] = ClipGrad(dClassifierLogits[k]);
        }

        float dHidden[classifierHiddenSize] = {0.0f};

        for(int k = 0; k < numClasses; k++){
            for(int j = 0; j < classifierHiddenSize; j++){
                dHidden[j] += dClassifierLogits[k] * classifierW2[cl][k][j];
            }
        }

        float dHiddenPre[classifierHiddenSize];

        for(int j = 0; j < classifierHiddenSize; j++){
            dHiddenPre[j] = dHidden[j] * ReLUGrad(classifierHiddenPre[cl][j]);
            dHiddenPre[j] = ClipGrad(dHiddenPre[j]);

            for(int m = 0; m < representationSize; m++){
                dRep[m] += dHiddenPre[j] * classifierW1[cl][j][m];
            }
        }

        for(int k = 0; k < numClasses; k++){
            classifierB2[cl][k] -= eta * dClassifierLogits[k];

            for(int j = 0; j < classifierHiddenSize; j++){
                classifierW2[cl][k][j] -= eta * dClassifierLogits[k] * classifierHidden[cl][j];
            }
        }

        for(int j = 0; j < classifierHiddenSize; j++){
            classifierB1[cl][j] -= eta * dHiddenPre[j];

            for(int m = 0; m < representationSize; m++){
                classifierW1[cl][j][m] -= eta * dHiddenPre[j] * rep[m];
            }
        }
    }

    for(int cl = 0; cl < numClassifiers; cl++){
        attentionB2[cl] -= eta * dAttentionLogits[cl];

        for(int j = 0; j < attentionHiddenSize; j++){
            attentionW2[cl][j] -= eta * dAttentionLogits[cl] * attentionHidden[j];
        }
    }

    for(int j = 0; j < attentionHiddenSize; j++){
        attentionB1[j] -= eta * dAttentionPre[j];

        for(int m = 0; m < representationSize; m++){
            attentionW1[j][m] -= eta * dAttentionPre[j] * rep[m];
        }
    }

    float dConvPre[numConvFilters][convTimeSteps];

    for(int f = 0; f < numConvFilters; f++){
        for(int t = 0; t < convTimeSteps; t++){
            dConvPre[f][t] = 0.0f;
        }
    }

    for(int f = 0; f < numConvFilters; f++){
        int maxT = maxIndex[f];

        float dMax = dRep[f];
        float dMean = dRep[f + numConvFilters];
        float dLast = dRep[f + 2 * numConvFilters];
        float dChange = dRep[f + 3 * numConvFilters];

        dConvPre[f][maxT] += dMax * ReLUGrad(convPre[f][maxT]);

        for(int t = 0; t < convTimeSteps; t++){
            dConvPre[f][t] += (dMean / (float)convTimeSteps) * ReLUGrad(convPre[f][t]);
        }

        int firstT = 0;
        int lastT = convTimeSteps - 1;

        dConvPre[f][lastT] += dLast * ReLUGrad(convPre[f][lastT]);
        dConvPre[f][lastT] += dChange * ReLUGrad(convPre[f][lastT]);
        dConvPre[f][firstT] -= dChange * ReLUGrad(convPre[f][firstT]);
    }

    for(int f = 0; f < numConvFilters; f++){
        float biasGrad = 0.0f;
        float weightGrad[convKernelSize][inputSize];

        for(int r = 0; r < convKernelSize; r++){
            for(int m = 0; m < inputSize; m++){
                weightGrad[r][m] = 0.0f;
            }
        }

        for(int t = 0; t < convTimeSteps; t++){
            float grad = ClipGrad(dConvPre[f][t]);
            biasGrad += grad;

            for(int r = 0; r < convKernelSize; r++){
                for(int m = 0; m < inputSize; m++){
                    weightGrad[r][m] += grad * seq[t + r][m];
                }
            }
        }

        convB[f] -= eta * ClipGrad(biasGrad);

        for(int r = 0; r < convKernelSize; r++){
            for(int m = 0; m < inputSize; m++){
                convW[f][r][m] -= eta * ClipGrad(weightGrad[r][m]);
            }
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

float MiniCABClassifier::ReLU(float x) const{
    if(x > 0.0f){
        return x;
    }

    return 0.01f * x;
}

float MiniCABClassifier::ReLUGrad(float x) const{
    if(x > 0.0f){
        return 1.0f;
    }

    return 0.01f;
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
        values[i] = std::exp(values[i] - maxVal);
        sum += values[i];
    }

    if(sum > 0.0f){
        for(int i = 0; i < n; i++){
            values[i] /= sum;
        }
    }
}