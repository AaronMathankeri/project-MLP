/**
 *   \file ioFunctions.cpp
 *   \brief source fo ioFunctions.hpp
 *
 *  Detailed description
 *
 */
#include "ioFunctions.hpp"
//-----------------------------------------------------
//I-O functions
void printMatrix( float *Matrix , int nRows, int nCols ){
      for ( int i = 0; i < nRows ; i++) {
	    for ( int j = 0; j < nCols ; j++) {
		  printf ("%12.5f", Matrix[i*nCols + j]);
	    }
	    printf ("\n");
      }
}
