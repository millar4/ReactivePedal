#pragma once
#include "FeatureExtractor.h"

struct CABOutput{
    float scores[4];
    int predictedClass;
};

class MiniCABClassifier{
    private:
        static constexpr int numClasses = 4;
        static constexpr int numClassifiers = 4;
        static constexpr int timeSteps = 32;
        static constexpr int inputSize = 12;

        static constexpr int numConvFilters = 12;
        static constexpr int convKernelSize = 5;
        static constexpr int convTimeSteps = timeSteps - convKernelSize + 1;

        static constexpr int representationParts = 4;
        static constexpr int sequenceStatsSize = 16;
        static constexpr int representationSize = numConvFilters * representationParts + sequenceStatsSize;

        static constexpr int classifierHiddenSize = 20;
        static constexpr int attentionHiddenSize = 20;

    public:
        void Init();
        void Reset();

        void PushFrame(const AudioFeatures& f);
        CABOutput PredictSequence() const;

        void TrainFromExample(const AudioFeatures& f, int label, float eta);
        void TrainFromCurrentSequence(int label, float eta);

        void SetNormalisation(const float mean[inputSize], const float std[inputSize]);

    private:
        AudioFeatures frameBuffer[timeSteps];
        int frameIndex = 0;
        int frameCount = 0;

        float convW[numConvFilters][convKernelSize][inputSize];
        float convB[numConvFilters];

        float classifierW1[numClassifiers][classifierHiddenSize][representationSize];
        float classifierB1[numClassifiers][classifierHiddenSize];
        float classifierW2[numClassifiers][numClasses][classifierHiddenSize];
        float classifierB2[numClassifiers][numClasses];

        float attentionW1[attentionHiddenSize][representationSize];
        float attentionB1[attentionHiddenSize];
        float attentionW2[numClassifiers][attentionHiddenSize];
        float attentionB2[numClassifiers];

        float normMean[inputSize];
        float normStd[inputSize];
        bool hasNormalisation = false;

        void FeaturesToInput(const AudioFeatures& f, float input[inputSize]) const;

        void BuildSequenceFromBuffer(float seq[timeSteps][inputSize]) const;
        void BuildSequenceWithLatest(const AudioFeatures& f, float seq[timeSteps][inputSize]) const;

        void CNNForward(
            const float seq[timeSteps][inputSize],
            float convPre[numConvFilters][convTimeSteps],
            float convAct[numConvFilters][convTimeSteps],
            float rep[representationSize],
            int maxIndex[numConvFilters]
        ) const;

        void ClassifierPredict(
            int classifierId,
            const float rep[representationSize],
            float probs[numClasses]
        ) const;

        void AttentionUnitPredict(
            const float rep[representationSize],
            float alpha[numClassifiers]
        ) const;

        void TrainOnSequence(const float seq[timeSteps][inputSize], int label, float eta);

        float ReLU(float x) const;
        float ReLUGrad(float x) const;
        void Softmax(float* values, int n) const;
};