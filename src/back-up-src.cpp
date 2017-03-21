      /*
      // test MLP functions
      // cross entropy function
      y[0] = 0.5;
      cout << "Entropy of system is " << crossEntropyFunction( t , y ) << "\n";

      float * test = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 );
      initializeMatrix( test, 1 , NUM_SAMPLES);
	    
      logisticSigmoid( t, test, NUM_OUTPUTS );
      cout <<"Input of sigmoid " << t[0] << "\n";
      cout <<"Output of Logistic Sigmoid is " << test[0] << "\n";

      initializeMatrix( test, 1 , NUM_SAMPLES);

      dlogisticSigmoid( t, test, NUM_OUTPUTS );
      cout <<"Input of dSigmoid " << t[0] << "\n";
      cout <<"Output of dLogistic Sigmoid is " << test[0] << "\n";
      printf("-------------------------------------\n");
      */
      //--------------------------------------------------------------------

float * activationFunction( float *a ){
      //sigma(a) = 1/(1 + exp(-a))

      float * z = (float *)mkl_malloc( NUM_HIDDEN_NODES * sizeof( float ), 64 );
      initializeMatrix( z , 1 , NUM_HIDDEN_NODES);

      //multiply by -1
      for (int i = 0; i < (NUM_HIDDEN_NODES); i++) {
	    a[i] = a[i] * (-1);
      }

      vsExp( (NUM_HIDDEN_NODES), a, a );

      //compute activation!
      for (int i = 0; i < (NUM_HIDDEN_NODES); i++) {
	    z[i] = 1/( 1 + a[i]);
      }
      return z;
}

void errorFunction( float *t, float *finalOutputs){
      cout << "Error function for training" << endl;
      float * diff = (float *)mkl_malloc( NUM_SAMPLES * sizeof( float ), 64 );
      float res = 0.0;
      const int incx = 1;
      
      for (int i = 0; i < (NUM_SAMPLES); i++) {
	    diff[i] = (float)(0.0);
      }
      //subtract them!
      vsSub( NUM_SAMPLES , t , finalOutputs, diff);
      //get norm
      res = snrm2( &(NUM_SAMPLES), diff, &incx);
      //get norm squared
      res *= res;
      //multiply by -1.0
      res = res * -1.0;
      cout << "Error = " << res << endl;
}

//float * x = (float *)mkl_malloc( NUM_SAMPLES* NUM_FEATURES*sizeof( float ), 64 );
//initializeMatrix( x, NUM_SAMPLES, NUM_FEATURES );

/*
  x[2]	=  1.65755662;  x[3]  = -0.63203157;
  x[4]	= -0.15878875;  x[5]  =  0.25584465;
  x[6]	= -1.088752  ;  x[7]  = -0.39694315;
  x[8]	=  1.768052  ;  x[9]  = -0.25443213;
  x[10]	=  1.95416454;  x[11] = -0.12850579;
  x[12]	=  0.93694537;  x[13] =  0.36597075;
  x[14]	=  0.88446589;  x[15] = -0.47595401;
  x[16]	=  0.80950246;  x[17] =  0.3505231 ;
  x[18]	=  1.2278091;   x[19] = -0.64785108;
*/
/*
  t[1] = 1; t[6] = 0;
  t[2] = 1; t[7] = 1;
  t[3] = 0; t[8] = 0;
  t[4] = 1; t[9] = 1;
*/

