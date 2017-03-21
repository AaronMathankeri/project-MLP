/**
 *   \file gradientDescent.hpp
 *   \brief gradient descent related algorithms
 *
 *  Detailed description
 *
 */
#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <iostream>

#include "mlpParameters.hpp"
#include "ioFunctions.hpp"

using namespace std;

//-----------------------------------------------------
// Gradient descent updates
void updateFirstLayerWeights( float* firstLayerWeightMatrix, const float* firstLayerDerivatives, const float LEARNING_RATE);

void updateSecondLayerWeights( float* secondLayerWeightVector, const float* secondLayerDerivatives, const float LEARNING_RATE);

//-----------------------------------------------------

#endif /* GRADIENTDESCENT_H */
