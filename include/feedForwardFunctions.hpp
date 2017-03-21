/**
 *   \file feedForwardFunctions.hpp
 *   \brief declare feed forward network functions
 *
 *  Detailed description
 *
 */

#ifndef FEEDFORWARDFUNCTIONS_H
#define FEEDFORWARDFUNCTIONS_H

#include <iostream>
#include "mkl.h"
#include "mlpParameters.hpp"
#include "networkAgnosticFunctions.hpp"
using namespace std;

//-----------------------------------------------------
// Feed-forward Functions
void computeActivations( float* x, float* firstLayerWeightMatrix, float* a);
void computeHiddenUnits( float* a, float* z, int length);
void computeOutputActivations( float* z, float* secondLayerWeightVector, float* v);
//-----------------------------------------------------
//void logisticSigmoid( float * a , float *sigma, int length);

#endif /* FEEDFORWARDFUNCTIONS_H */
