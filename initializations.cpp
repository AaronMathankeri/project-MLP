/**
 *   \file initializations.cpp
 *   \brief source for initializations.hpp
 *
 *  Detailed description
 *
 */

#include "initializations.hpp"

//-----------------------------------------------------
//initialize to avoid memory errors
void initializeMatrix( float * Matrix, int rows, int columns ){
      memset( Matrix, 0.0,  rows*columns * sizeof(double));      
}
float fRand(float fMin, float fMax){
      float f = (float)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}
void setRandomWeights( float * weights, int nRows, int nCols ){
      for (int i = 0; i < (nRows*nCols); ++i) {
	    float temp = fRand( -10.0, 10.0);
	    weights[i] = temp;
      }
}
