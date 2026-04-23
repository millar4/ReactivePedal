//keep our previous waveform and feature extractor
//stop using final phrase vector
// short sequence of recent feature frames
// run tiny classifiers on each frame 
//small attention unit to weight them
//then average results over time 


//establish our weights and our biases
//classifierHiddenSize is our hidden layer

void MiniCABClassifier::Init(){
    for(int cl = 0; cl < numClassifiers; cl++){
        for(int i = 0; i < classifierHiddenSize; i++){
            classifierB1[cl][i] = 0.0f; 
            for(int j = 0; j < inputSize; j++){
                // aka the weight connecting input feature j hidden neuron i in classifier cl
                classifierW1[cl][i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            }
        }
        //for second weights
        for(int i = 0; i < numClasses; i++){
            classifierB2[cl][i] = 0.0f;
            for(int j = 0; j < classifierHiddenSize; j++){
                classifierW2[cl][i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            }
        }
    }

    for(int i = 0; i < attentionHiddenSize; i++){
        attentionB1[i] = 0.0f;
        for(int j = 0; j < inputSize; j++){
            attentionW1[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    for(int i = 0; i < numClassifiers; i++){
        attentionB2[i] = 0.0f;
        for(int j = 0; j < attentionHiddenSize; j++){
            attentionW2[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    Reset();
}


//we clear the stored sequence state
void MiniCABClassifier::Reset(){
    frameIndex = 0;
     //write the frames from the beginning of the buffer
    frameCount = 0;
    //zero out features, so we have an empty buffer when further processing
    std::memset(frameBuffer, 0, sizeof(frameBuffer));
}

void MiniCABClassifier::PushFrame(const AudioFeatures& f){
    //newest frame feature into current slot
    frameBuffer[frameIndex] = f;
    //move to next slot, wrap around if reaching end
    frameIndex = (frameIndex + 1) % timeSteps;
    //keep track of current time slots
     if(frameCount < timeSteps){
        frameCount++;
    }
    //
}

CABOutput MiniCABClassifier::PredictSequence() const{
    //create an empty output struct
    CABOutput out = {};
    float finalScores[numClasses] = {0.0f};

    //early stop 
    if(frameCount == 0){
        return out;
    }

    //convert latest stored frame into plain input array
    float input[inputSize];
    int lastIndex = (frameIndex - 1 + timeSteps) % timeSteps;
    FeaturesToInput(frameBuffer[lastIndex], input);

    //get our alpha
    float alpha[numClassifiers];
    AttentionUnitPredict(input, alpha);


    //run the classifiers
    for(int cl = 0; cl < numClassifiers; cl++){
        float probs[numClasses];
        ClassifierPredict(cl, input, probs);
        //weight by alpha
        for(int k = 0; k < numClasses; k++){
            finalScores[k] += alpha[cl] * probs[k];
        }
    }

    Softmax(finalScores, numClasses);

    int bestClass = 0;
    float bestScore = finalScores[0];

    //pick the best class
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
    input[10] = 0.0f;
    input[11] = 0.0f;
}

void MiniCABClassifier::ClassifierPredict(int classifierId, const float input[inputSize], float probs[numClasses]) const{
    float hidden[classifierHiddenSize];

    for(int i = 0; i < classifierHiddenSize; i++){
        float sum = classifierB1[classifierId][i];
        for(int j = 0; j < inputSize; j++){
            sum += classifierW1[classifierId][i][j] * input[j];
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