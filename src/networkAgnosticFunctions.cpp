/**
 *   \file networkAgnosticFunctions.cpp
 *   \brief source code
 *
 *  Detailed description
 *
 */

#include "networkAgnosticFunctions.hpp"

//-----------------------------------------------------
// MLP network Independent functions
float crossEntropyFunction( float *t, float *y ){
      float entropy = 0.0;
      for (int i = 0; i < NUM_SAMPLES; ++i){
	    entropy += -(t[i]*log(y[i]) + (1 - t[i])*log(1 - y[i]));		  
      }
      return entropy;
}

void logisticSigmoid( float * a , float *sigma, int length){
      //sigma(a) = 1/(1 + exp(-a))
      for (int i = 0; i < (length); i++) {
	    sigma[i] = 1/( 1 + exp(-a[i]));
      }
}

void dlogisticSigmoid( float * a , float * sigmaPrime, int length){
      //d/da sigma(a) = sigma(a) * [1 - sigma(a)]
      logisticSigmoid(a, sigmaPrime, length);
      for (int i = 0; i < (length); ++i) {
	    sigmaPrime[i] = sigmaPrime[i]*(1 - sigmaPrime[i] );
      }
}

