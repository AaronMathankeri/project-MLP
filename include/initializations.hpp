/**
 *   \file initializations.hpp
 *   \brief to avoid memory errors and quickly initialize values
 *
 *  Detailed description
 *
 */
#ifndef INITIALIZATIONS_H
#define INITIALIZATIONS_H

#include <iostream>

using namespace std;

//-----------------------------------------------------
//initialize to avoid memory errors
void initializeMatrix( float * Matrix, int rows, int columns );
float fRand(float fMin, float fMax);
void setRandomWeights( float * weights, int nRows, int nCols );

#endif /* INITIALIZATIONS_H */
