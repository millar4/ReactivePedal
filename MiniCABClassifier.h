#pragma once
#include "FeatureExtractor.h"

struct CABOutput{
    float scores[4];
    int predictedClass;
};

class MiniCABClassifier{
    private:
        static constexpr int numClasses = 4;
        static constexpr int timeSteps = 256;
        static constexpr int inputSize = 12;

        static constexpr int numConvFilters = 8;
        static constexpr int convKernelSize = 5;
        static constexpr int convTimeSteps = timeSteps - convKernelSize + 1;

        static constexpr int representationParts = 3;
        static constexpr int temporalStatsSize = 40;
        static constexpr int representationSize = numConvFilters * representationParts + temporalStatsSize;

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

        float classifierW[numClasses][representationSize];
        float classifierB[numClasses];

        float normMean[inputSize];
        float normStd[inputSize];
        bool hasNormalisation = false;

        void FeaturesToInput(const AudioFeatures& f, float input[inputSize]) const;

        void BuildSequenceFromBuffer(float seq[timeSteps][inputSize]) const;
        void BuildSequenceWithLatest(const AudioFeatures& f, float seq[timeSteps][inputSize]) const;

        void BuildRepresentation(
            const float seq[timeSteps][inputSize],
            float rep[representationSize]
        ) const;

        void ClassifierPredict(
            const float rep[representationSize],
            float probs[numClasses]
        ) const;

        void TrainOnSequence(const float seq[timeSteps][inputSize], int label, float eta);

        float ClipValue(float x, float lo, float hi) const;
        float ConvFilterValue(const float seq[timeSteps][inputSize], int filterId, int t) const;
        float EventAttentionLogit(const float input[inputSize]) const;
        float GapAttentionLogit(const float input[inputSize]) const;
        void Softmax(float* values, int n) const;
};
