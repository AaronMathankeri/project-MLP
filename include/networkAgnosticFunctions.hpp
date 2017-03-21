/**
 *   \file networkAgnosticFunctions.hpp
 *   \brief important functions for mlp
 *
 *  Detailed description
 * functions independent of network yet critical
 */
#ifndef NETWORKAGNOSTICFUNCTIONS_H
#define NETWORKAGNOSTICFUNCTIONS_H

#include "mathimf.h"
#include "mlpParameters.hpp"
//-----------------------------------------------------
// MLP network Independent functions
float crossEntropyFunction( float *t, float *y );
void logisticSigmoid( float * a , float *sigma, int length);
void dlogisticSigmoid( float * a , float * sigmaPrime, int length);

#endif /* NETWORKAGNOSTICFUNCTIONS_H */
