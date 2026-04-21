#include "NeuralNet.h"
#include <cmath>
#include <cstdlib>

void NeuralNet::Init(){
    for(int i = 0; i < inputSize; i++){
        mean_[i] = 0.0f;
        std_[i]  = 1.0f;
    }

    for(int i = 0; i < hiddenSize; i++){
        b1_[i] = 0.0f;
        for(int j = 0; j < inputSize; j++){
            W1_[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    for(int i = 0; i < outputSize; i++){
        b2_[i] = 0.0f;
        for(int j = 0; j < hiddenSize; j++){
            W2_[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }
}

void NeuralNet::Normalisation(const float mean[12], const float std[12]){
    for(int i = 0; i < inputSize; i++){
        mean_[i] = mean[i];
        std_[i]  = std[i];
        if(std_[i] < 1e-6f){
            std_[i] = 1.0f;
        }
    }
}

void NeuralNet::NormaliseInput(const AudioFeatures& f, float input[inputSize]) const{
    float raw[inputSize] = {
        f.rms,
        f.peak,
        f.zcr,
        f.spectralCentroid,
        f.spectralFlux,
        f.rmsDelta,
        f.envelope,
        f.envelopeDelta,
        f.rmsVariance,
        f.centroidVariance,
        f.onsetCount,
        f.timeSinceLastOnset
    };

    for(int i = 0; i < inputSize; i++){
        input[i] = (raw[i] - mean_[i]) / std_[i];
    }
}

float NeuralNet::ReLU(float x) const{
    return x > 0.0f ? x : 0.0f;
}

NNOutput NeuralNet::Predict(const AudioFeatures& features) const{
    float input[inputSize];
    float hidden[hiddenSize];
    float hiddenPre[hiddenSize];
    float output[outputSize];

    NormaliseInput(features, input);

    for(int i = 0; i < hiddenSize; i++){
        float sum = b1_[i];
        for(int j = 0; j < inputSize; j++){
            sum += W1_[i][j] * input[j];
        }
        hiddenPre[i] = sum;
        hidden[i] = ReLU(sum);
    }

    for(int i = 0; i < outputSize; i++){
        float sum = b2_[i];
        for(int j = 0; j < hiddenSize; j++){
            sum += W2_[i][j] * hidden[j];
        }
        output[i] = sum;
    }

    float maxLogit = output[0];
    for(int i = 1; i < outputSize; i++){
        if(output[i] > maxLogit){
            maxLogit = output[i];
        }
    }

    float sumExp = 0.0f;
    float probs[outputSize];

    for(int i = 0; i < outputSize; i++){
        probs[i] = expf(output[i] - maxLogit);
        sumExp += probs[i];
    }

    int best = 0;
    float bestVal = 0.0f;

    for(int i = 0; i < outputSize; i++){
        probs[i] /= sumExp;
        if(probs[i] > bestVal){
            bestVal = probs[i];
            best = i;
        }
    }

    NNOutput out;
    for(int i = 0; i < outputSize; i++){
        out.scores[i] = probs[i];
    }
    out.predictedClass = best;

    return out;
}

void NeuralNet::Train(const AudioFeatures& features, int targetClass, float eta){
    float input[inputSize];
    float hidden[hiddenSize];
    float hiddenPre[hiddenSize];
    float output[outputSize];

    NormaliseInput(features, input);

    for(int i = 0; i < hiddenSize; i++){
        float sum = b1_[i];
        for(int j = 0; j < inputSize; j++){
            sum += W1_[i][j] * input[j];
        }
        hiddenPre[i] = sum;
        hidden[i] = ReLU(sum);
    }

    for(int i = 0; i < outputSize; i++){
        float sum = b2_[i];
        for(int j = 0; j < hiddenSize; j++){
            sum += W2_[i][j] * hidden[j];
        }
        output[i] = sum;
    }

    float maxLogit = output[0];
    for(int i = 1; i < outputSize; i++){
        if(output[i] > maxLogit){
            maxLogit = output[i];
        }
    }

    float probs[outputSize];
    float sumExp = 0.0f;

    for(int i = 0; i < outputSize; i++){
        probs[i] = expf(output[i] - maxLogit);
        sumExp += probs[i];
    }

    for(int i = 0; i < outputSize; i++){
        probs[i] /= sumExp;
    }

    float dOut[outputSize];
    for(int i = 0; i < outputSize; i++){
        dOut[i] = probs[i];
    }
    dOut[targetClass] -= 1.0f;

    // Update W2
    for(int i = 0; i < outputSize; i++){
        for(int j = 0; j < hiddenSize; j++){
            W2_[i][j] -= eta * dOut[i] * hidden[j];
        }
        b2_[i] -= eta * dOut[i];
    }

    float dHidden[hiddenSize];

    for(int j = 0; j < hiddenSize; j++){
        float grad = 0.0f;
        for(int i = 0; i < outputSize; i++){
            grad += dOut[i] * W2_[i][j];
        }
        dHidden[j] = (hiddenPre[j] > 0.0f) ? grad : 0.0f;
    }

    for(int i = 0; i < hiddenSize; i++){
        for(int j = 0; j < inputSize; j++){
            W1_[i][j] -= eta * dHidden[i] * input[j];
        }
        b1_[i] -= eta * dHidden[i];
    }
}

void NeuralNet::SaveState(MappingState& state) const{
    for(int i = 0; i < inputSize; i++){
        state.mean[i] = mean_[i];
        state.std[i]  = std_[i];
    }

    for(int i = 0; i < hiddenSize; i++){
        state.b1[i] = b1_[i];
        for(int j = 0; j < inputSize; j++){
            state.W1[i][j] = W1_[i][j];
        }
    }

    for(int i = 0; i < outputSize; i++){
        state.b2[i] = b2_[i];
        for(int j = 0; j < hiddenSize; j++){
            state.W2[i][j] = W2_[i][j];
        }
    }
}

void NeuralNet::LoadState(const MappingState& state){
    for(int i = 0; i < inputSize; i++){
        mean_[i] = state.mean[i];
        std_[i]  = state.std[i];
    }

    for(int i = 0; i < hiddenSize; i++){
        b1_[i] = state.b1[i];
        for(int j = 0; j < inputSize; j++){
            W1_[i][j] = state.W1[i][j];
        }
    }

    for(int i = 0; i < outputSize; i++){
        b2_[i] = state.b2[i];
        for(int j = 0; j < hiddenSize; j++){
            W2_[i][j] = state.W2[i][j];
        }
    }
}