#ifndef MCMC_STRETCH_CL
#define MCMC_STRETCH_CL

/**
 * Author = Robbert Harms
 * Date = 2014-02-01
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

 /*
 Original license:

Copyright (c) 2013, Alex Kaiser
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer. Redistributions
 in binary form must reproduce the above copyright notice, this list
 of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * Required compile time definitions.
 *
 * #define NMR_PARAMS                     | as the number of parameters to fit (n), the dimension of the program
 * #define EVAL_FUNC_NAME                 | the name of the eval function the routine should call
 * #define K_OVER_TWO                     | Number of walkers in each group
 * #define A_COEFF_0                      | the first component of the a parameter
 * #define A_COEFF_1                      | the second component of the a parameter
 * #define A_COEFF_2                      | the third component of the a parameter
 */

void mcmc_stretch(
    __global float *X_moving,                  // walkers to be updated
    __global float *log_prob_moving,           // cached log probabilities of the moving walkers, will be updated
    __global const float *X_fixed,             // fixed walkers
    __global float4 *ranluxcltab,              // state information for random number generator
    __global unsigned long *accepted,          // number of samples accepted
    const void *data,                          // data or observations
    const float beta){

    // start up data structures for the random number generator
    // ranluxclstate is a struct of 7 * total_work_items float4 variables
    // storing the state of the generator.
    ranluxcl_state_t ranluxclstate;

    //Download state into ranluxclstate struct.
    ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

    // get indexing information
    int lid             = get_local_id(0);
    int k               = get_global_id(0);

    // allocate for proposal
    #ifdef USE_LOCAL_PROPOSAL
        // allocate local if size permits
        __local float Y[NMR_PARAMS * WORK_GROUP_SIZE];

        // the first index owned by this work item in the local arrays
        const int start_idx = NMR_PARAMS * lid;
    #else
        // allocate into private, likely spills into global
        float Y[NMR_PARAMS];
        const int start_idx = 0;
    #endif

    // temps
    float z, q, log_py, log_pxk;
    int j;

    // random numbers go here
    float4 xi;

    // if we somehow start more work items than there are walkers in each group, move on
    if(k < K_OVER_TWO){

        // generate the three needed random numbers
        xi = ranluxcl(&ranluxclstate);

        // draw the walker randomly
        j = (int) (xi.s0 * K_OVER_TWO);

        // draw a sample from the g(Z) distribution
        z = A_COEFF_2 * xi.s1*xi.s1 + A_COEFF_1 * xi.s1 + A_COEFF_0;

        // compute the proposal
        for(int i=0; i<NMR_PARAMS; i++)
            Y[i + start_idx] = X_fixed[i + j*NMR_PARAMS] + z * (X_moving[i + k*NMR_PARAMS] - X_fixed[i + j*NMR_PARAMS]);

        // evaluate the likelihood function
        log_py  = calculateLogPDF(data, Y + start_idx);
        log_pxk = log_prob_moving[k];

        if(isinf(log_py)){
            // always reject an inf sample
            q = 0.0f;
        }
        else{
            // standard case
            q   = pown(z, NMR_PARAMS-1) * exp( beta * (log_py - log_pxk)) ;
        }

        // accept and update
        if(xi.s2 <= q){
            accepted[k]++ ;
            for(int i=0; i<NMR_PARAMS; i++){
                X_moving[i + k*NMR_PARAMS] = Y[i + start_idx];
            }
            log_prob_moving[k] = log_py;
        }
    }

    //Upload state again so that we don't get the same
    //numbers over again the next time we use ranluxcl.
    ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
}

#endif // MCMC_STRETCH_CL