/**
 *   \file feedForwardFunctions.cpp
 *   \brief source for functions
 *
 *  Detailed description
 *
 */
#include "feedForwardFunctions.hpp"
#include "mathimf.h"

//-----------------------------------------------------
// Feed-forward Functions
void computeActivations( float* x, float* firstLayerWeightMatrix, float* a){
      //1. get activations: a_j = \sum_i^D{w_ji * x_i + w_j0}
      // perform matrix vector multiplication: a = W*x
      cout << "Computing 1st Layer Activations" << "\n";
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      cblas_sgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, NUM_FEATURES,
		   alpha, firstLayerWeightMatrix, NUM_FEATURES, x, incx, beta, a, incx);
}

void computeHiddenUnits( float* a, float* z, int length){
      logisticSigmoid( a , z , length);
}

void computeOutputActivations( float* z, float* secondLayerWeightVector, float* v){
      // perform matrix vector multiplication: y = W*z
      cout << "Computing 2nd Layer Activations" << "\n";
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      float res = 0.0;
      res = cblas_sdot( NUM_HIDDEN_NODES, z,incx, secondLayerWeightVector, incx);
      v[(NUM_OUTPUTS - 1)] = res;
}
//-----------------------------------------------------

