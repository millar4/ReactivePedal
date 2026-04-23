#pragma once
#include "FeatureExtractor.h"

struct CABOutput{
    float scores[4];
    int predictedClass;
};

class MiniCABClassifier{
    public:
        void Init();
        void Reset();

        void PushFrame(const AudioFeatures& f);
        CABOutput PredictSequence() const;

    private:
        static constexpr int numClasses = 4;
        static constexpr int numClassifiers = 3;
        static constexpr int timeSteps = 6;
        static constexpr int inputSize = 12;
        static constexpr int classifierHiddenSize = 8;
        static constexpr int attentionHiddenSize = 8;

        AudioFeatures frameBuffer[timeSteps];
        int frameIndex = 0;
        int frameCount = 0;

        float classifierW1[numClassifiers][classifierHiddenSize][inputSize];
        float classifierB1[numClassifiers][classifierHiddenSize];
        float classifierW2[numClassifiers][numClasses][classifierHiddenSize];
        float classifierB2[numClassifiers][numClasses];

        float attentionW1[attentionHiddenSize][inputSize];
        float attentionB1[attentionHiddenSize];
        float attentionW2[numClassifiers][attentionHiddenSize];
        float attentionB2[numClassifiers];

        void FeaturesToInput(const AudioFeatures& f, float input[inputSize]) const;
        void ClassifierPredict(int classifierId, const float input[inputSize], float probs[numClasses]) const;
        void AttentionUnitPredict(const float input[inputSize], float alpha[numClassifiers]) const;

        float ReLU(float x) const;
        void Softmax(float* values, int n) const;
};