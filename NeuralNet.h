#pragma once

#include "FeatureExtractor.h"

enum Style{
    STYLE_DISTORTION = 0,
    STYLE_CHORUS,
    STYLE_AMBIENT,
    STYLE_REVERB,
    STYLE_COUNT
};

struct NNOutput{
    float scores[STYLE_COUNT];
    int predictedClass;
};

struct MappingState{
    float mean[12];
    float std[12];
    float W1[8][12];
    float b1[8];
    float W2[4][8];
    float b2[4];
};

class NeuralNet{
    public:
    void Init();
    void Normalisation(const float mean[12], const float std[12]);
    NNOutput Predict(const AudioFeatures& features) const;
    void Train(const AudioFeatures& features, int targetClass, float eta);
    void SaveState(MappingState& state) const;
    void LoadState(const MappingState& state);

    private:
        static constexpr int inputSize = 12;
        static constexpr int hiddenSize = 8;
        static constexpr int outputSize = 4;

        float mean_[inputSize];
        float std_[inputSize];

        float W1_[hiddenSize][inputSize];
        float b1_[hiddenSize];

        float W2_[outputSize][hiddenSize];
        float b2_[outputSize];

        void NormaliseInput(const AudioFeatures& features, float input[inputSize]) const;
        float ReLU(float x) const;
};