#pragma once

void ScreenInit();
void ScreenFillRed();
void ScreenFillBlack();

void ScreenDrawHome(const char* presetName,
                    const char* networkName,
                    int isTraining,
                    int predictionMode,
                    int predictedClass,
                    int stableClass);

void ScreenDrawMenu(int selectedIndex,
                    const char* presetName,
                    const char* networkName,
                    int isTraining,
                    int predictionMode,
                    int hasSavedMapping,
                    int b0,
                    int b1,
                    int b2,
                    int b3);