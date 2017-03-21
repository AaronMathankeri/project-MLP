/**
 *   \file backpropFunctions.cpp
 *   \brief the source
 *
 *  Detailed description
 *
 */
#include "backpropFunctions.hpp"
//-----------------------------------------------------
// BackProp algorithms
float computeOutputErrors( float* y, float* t){
      float delta = 0.0;
      delta = y[(NUM_OUTPUTS - 1)] - t[(NUM_OUTPUTS - 1)];
      return delta;
}
void computeHiddenErrors( float* a, float* secondLayerWeightVector, float delta, float* hiddenDeltas){
      cout << "Computing Hidden Errors" << endl;
      float * dA = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      float * temp = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      initializeMatrix( dA, NUM_HIDDEN_NODES, 1);
      initializeMatrix( temp, NUM_HIDDEN_NODES, 1);
      
      dlogisticSigmoid( a, dA, NUM_HIDDEN_NODES );
      for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
	    temp[i] = secondLayerWeightVector[i]*delta;
	    hiddenDeltas[i] = dA[i]*temp[i];
      }
}
void computeSecondLayerDerivatives( float* z, float delta, float* secondLayerDerivatives){
      for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
	    secondLayerDerivatives[i] += z[i]*delta;
      }
}
void computeFirstLayerDerivatives( float* x, float* hiddenDeltas, float* firstLayerDerivatives){
      for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
	    for (int j = 0; j < NUM_FEATURES; ++j) {
		  firstLayerDerivatives[i*NUM_FEATURES + j] += hiddenDeltas[i]*x[j];
	    }
      }
}
//-----------------------------------------------------
