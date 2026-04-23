//keep our previous waveform and feature extractor
//stop using final phrase vector
// short sequence of recent feature frames
// run tiny classifiers on each frame 
//small attention unit to weight them
//then average results over time 


//establish our weights and our biases
//
void MiniCABClassifier::Init(){
    for(int cl = 0; cl < numClassifiers; c++){
        for(int i = 0; i < classifierHiddenSize; i++){
            classifierB1[cl][i] = 0.0f; 
            for(int j = 0; j < inputSize; j++){
                // aka the weight connecting input feature j hidden neuron i in classifier cl
                classifierW1[cl][i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            }
        }

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


