/**
 *   \file backpropFunctions.hpp
 *   \brief backpropagation functions
 *
 *  Detailed description
 *
 */

#ifndef BACKPROPFUNCTIONS_H
#define BACKPROPFUNCTIONS_H

#include <iostream>
#include "mkl.h"

#include "mlpParameters.hpp"
#include "initializations.hpp"
#include "networkAgnosticFunctions.hpp"

using namespace std;

//-----------------------------------------------------
// BackProp algorithms
float computeOutputErrors( float* y, float* t);
void computeHiddenErrors( float* a, float* secondLayerWeightVector, float delta, float* hiddenDeltas);
void computeSecondLayerDerivatives( float* z, float delta, float* secondLayerDerivatives);
void computeFirstLayerDerivatives( float* x, float* hiddenDeltas, float* firstLayerDerivatives);
//-----------------------------------------------------

#endif /* BACKPROPFUNCTIONS_H */
