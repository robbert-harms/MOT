#ifndef LMMIN_CL
#define LMMIN_CL

/*
 * Library:   lmfit (Levenberg-Marquardt least squares fitting)
 * File:      lmmin.c
 * Contents:  Levenberg-Marquardt minimization.
 * Copyright: MINPACK authors, The University of Chikago (1980-1999)
 *            Joachim Wuttke, Forschungszentrum Juelich GmbH (2004-2013)
 *            Robbert Harms (2013)
 *            Updated to version 7.1 (2018)
 * License:   see ../COPYING (FreeBSD)
 * Homepage:  apps.jcns.fz-juelich.de/lmfit

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

/**
 * Adapted by = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/** The evaluation function we are expecting. */
void %(FUNCTION_NAME)s(local mot_float_type* x, void* data_void, local mot_float_type* result);

/* function declarations. */
void lm_lmpar( const int n,
               local mot_float_type * const r,
               int ldr,
               local const int* const Pivot,
               local mot_float_type *const diag,
               local mot_float_type* const qtb,
               const mot_float_type delta,
               local mot_float_type * const par,
               local mot_float_type * const x,
               local mot_float_type * const Sdiag,
               local mot_float_type * const aux,
               local mot_float_type * const xdi );

void lm_qrfac( const int m,
               const int n,
               local mot_float_type * const A,
               local int* const Pivot,
               local mot_float_type* const Rdiag,
               local mot_float_type* const Acnorm,
               local mot_float_type* const W );

void lm_qrsolv( const int n,
                local mot_float_type * const r,
                const int ldr,
                local const int * const Pivot,
                local const mot_float_type * const diag,
                local const mot_float_type * const qtb,
                local mot_float_type * const x,
                local mot_float_type * const Sdiag,
                local mot_float_type * const W );

double lm_euclidian_norm(local const mot_float_type* const x, const int n);

/*****************************************************************************/
/*  Numeric constants                                                        */
/*****************************************************************************/

#define LM_MACHEP     MOT_EPSILON   /* resolution of arithmetic */
#define LM_DWARF      MOT_MIN       /* smallest nonzero number */
#define LM_SQRT_DWARF sqrt(MOT_MIN) /* square should not underflow */
#define LM_SQRT_GIANT sqrt(MOT_MAX) /* square should not overflow */
#define LM_USERTOL    (%(USERTOL_MULT)r * LM_MACHEP)  /* users are recommended to require this */

#define FTOL LM_USERTOL      /* Relative error desired in the sum of squares.
                             Termination occurs when both the actual and
                             predicted relative reductions in the sum of squares
                             are at most ftol. */
#define XTOL LM_USERTOL      /* Relative error between last two approximations.
                             Termination occurs when the relative error between
                             two consecutive iterates is at most xtol. */
#define GTOL LM_USERTOL      /* Orthogonality desired between fvec and its derivs.
                             Termination occurs when the cosine of the angle
                             between fvec and any column of the Jacobian is at
                             most gtol in absolute value. */
#define STEP_BOUND %(STEP_BOUND)r     /* Used in determining the initial step bound. This
                             bound is set to the product of stepbound and the
                             Euclidean norm of diag*x if nonzero, or else to
                             stepbound itself. In most cases stepbound should lie
                             in the interval (0.1,100.0). Generally, the value
                             100.0 is recommended. */
#define PATIENCE %(PATIENCE)r /* Used to set the maximum number of function evaluations
                             to patience*(number_of_parameters+1). */
#define SCALE_DIAG %(SCALE_DIAG)r  /* If 1, the variables will be rescaled internally.
                             Recommended value is 1. */
#define MAXFEV (PATIENCE * (%(NMR_PARAMS)s+1)) /** the maximum number of evaluations */

#define LM_ENORM_SQRT_GIANT LM_SQRT_GIANT /* square should not overflow */
#define LM_ENORM_SQRT_DWARF LM_SQRT_DWARF /* square should not underflow */

/**
 * Make sure that the following holds:
 * %(NMR_PARAMS)s > 0
 * %(NMR_OBSERVATIONS)s >= %(NMR_PARAMS)s
 * FTOL >= 0. && XTOL >= 0. && GTOL >= 0.
 * MAXFEV > 0
 * STEP_BOUND > 0.
 * SCALE_DIAG == 0 || SCALE_DIAG == 1
 */


/******************************************************************************/
/*  lmmin (main minimization routine)                                         */
/******************************************************************************/
int lmmin(local mot_float_type * const model_parameters, void* data,
          local mot_float_type* scratch_mot_float_type, local int* scratch_int){

    int j, i;
    int nfev = 0;
    bool outer_done_first = false;  /* loop flags, for monitoring */
    bool inner_done_first = false;
    bool inner_success; /* flag for loop control */
    double sum;
    mot_float_type delta = 0;
    mot_float_type actred, dirder, prered, ratio, tmp;

    local mot_float_type* scratch_ind = scratch_mot_float_type;

    local mot_float_type* temp1 = scratch_ind++;
    local mot_float_type* temp2 = scratch_ind++;
    local mot_float_type* xnorm = scratch_ind++;
    local mot_float_type* pnorm = scratch_ind++;
    local mot_float_type* fnorm = scratch_ind++;
    local mot_float_type* fnorm1 = scratch_ind++;
    local mot_float_type* gnorm = scratch_ind++;
    local mot_float_type* lmpar = scratch_ind++;
    local mot_float_type* fvec = scratch_ind;  scratch_ind += %(NMR_OBSERVATIONS)s;
    local mot_float_type* wf = scratch_ind;    scratch_ind += %(NMR_OBSERVATIONS)s;
    local mot_float_type* diag = scratch_ind;  scratch_ind += %(NMR_PARAMS)s;
    local mot_float_type* qtf = scratch_ind;   scratch_ind += %(NMR_PARAMS)s;
    local mot_float_type* wa1 = scratch_ind;   scratch_ind += %(NMR_PARAMS)s;
    local mot_float_type* wa2 = scratch_ind;   scratch_ind += %(NMR_PARAMS)s;
    local mot_float_type* wa3 = scratch_ind;   scratch_ind += %(NMR_PARAMS)s;
    local mot_float_type* fjac = scratch_ind;  scratch_ind += %(NMR_PARAMS)s * %(NMR_OBSERVATIONS)s;

    local int* Pivot = scratch_int;

    if(get_local_id(0) == 0){
        *lmpar = 0;
        *xnorm = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Initialize diag. */
    if (!SCALE_DIAG) {
        if(get_local_id(0) == 0){
            for (j = 0; j < %(NMR_PARAMS)s; j++)
                diag[j] = 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /***  Evaluate function at starting point and calculate norm.  ***/

    %(FUNCTION_NAME)s(model_parameters, data, fvec);
    nfev = 1;

    if(get_local_id(0) == 0){
        *fnorm = lm_euclidian_norm(fvec, %(NMR_OBSERVATIONS)s);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (!isfinite(*fnorm)) {
	    return 10; /* nan */
    } else if (*fnorm <= LM_DWARF) {
        return 1;
    }

    /***  The outer loop: compute gradient, then descend.  ***/

    while(true){
        /** Calculate the Jacobian. **/
        %(JACOBIAN_FUNCTION_NAME)s(model_parameters, data, fvec, fjac);

        /** Compute the QR factorization of the Jacobian. **/

        /* fjac is an m by n array. The upper n by n submatrix of fjac is made
         *   to contain an upper triangular matrix R with diagonal elements of
         *   nonincreasing magnitude such that
         *
         *         P^T*(J^T*J)*P = R^T*R
         *
         *         (NOTE: ^T stands for matrix transposition),
         *
         *   where P is a permutation matrix and J is the final calculated
         *   Jacobian. Column j of P is column Pivot(j) of the identity matrix.
         *   The lower trapezoidal part of fjac contains information generated
         *   during the computation of R.
         *
         * Pivot is an integer array of length n. It defines a permutation
         *   matrix P such that jac*P = Q*R, where jac is the final calculated
         *   Jacobian, Q is orthogonal (not stored), and R is upper triangular
         *   with diagonal elements of nonincreasing magnitude. Column j of P
         *   is column Pivot(j) of the identity matrix.
         */
        if(get_local_id(0) == 0){
            lm_qrfac(%(NMR_OBSERVATIONS)s, %(NMR_PARAMS)s, fjac, Pivot, wa1, wa2, wa3);
            /* return values are Pivot, wa1=rdiag, wa2=acnorm */

            /** Form Q^T * fvec, and store first n components in qtf. **/
            for (i = 0; i < %(NMR_OBSERVATIONS)s; i++){
                wf[i] = fvec[i];
            }

            for(j = 0; j < %(NMR_PARAMS)s; j++){
                tmp = fjac[j*%(NMR_OBSERVATIONS)s+j];
                if (tmp != 0) {
                    sum = 0;
                    for (i = j; i < %(NMR_OBSERVATIONS)s; i++){
                        sum += fjac[j*%(NMR_OBSERVATIONS)s+i] * wf[i];
                    }
                    tmp = -sum / tmp;
                    for (i = j; i < %(NMR_OBSERVATIONS)s; i++){
                        wf[i] += fjac[j*%(NMR_OBSERVATIONS)s+i] * tmp;
                    }
                }
                fjac[j*%(NMR_OBSERVATIONS)s+j] = wa1[j];
                qtf[j] = wf[j];
            }

            /**  Compute norm of scaled gradient and detect degeneracy. **/
            *gnorm = 0;
            for (j = 0; j < %(NMR_PARAMS)s; j++) {
                if(wa2[Pivot[j]] == 0){
                }
                else{
                    sum = 0;
                    for (i = 0; i <= j; i++){
                        sum += fjac[j*%(NMR_OBSERVATIONS)s+i] * qtf[i];
                    }
                    *gnorm = max((double)*gnorm, fabs(sum / wa2[Pivot[j]] / *fnorm));
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (*gnorm <= GTOL) {
            return 5;
        }

        /** Initialize or update diag and delta. **/
        if(!outer_done_first){ /* first iteration only */
            if(get_local_id(0) == 0){
                if (SCALE_DIAG) {
                    /* diag := norms of the columns of the initial Jacobian */
                    for (j = 0; j < %(NMR_PARAMS)s; j++){
                        diag[j] = 1;
                        if(wa2[j]){
                            diag[j] = wa2[j];
                        }
                    }
                    /* xnorm := || D x || */
                    for (j = 0; j < %(NMR_PARAMS)s; j++){
                        wa3[j] = diag[j] * model_parameters[j];
                    }
                    *xnorm = lm_euclidian_norm(wa3, %(NMR_PARAMS)s);
                } else {
                    *xnorm = lm_euclidian_norm(model_parameters, %(NMR_PARAMS)s);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(!isfinite(*xnorm)){
                return 10;
            }

            /* initialize the step bound delta. */
            if(*xnorm){
                delta = STEP_BOUND * *xnorm;
            }
            else{
                delta = STEP_BOUND;
            }
        } else {
            if(get_local_id(0) == 0){
                if (SCALE_DIAG) {
                    for (j = 0; j < %(NMR_PARAMS)s; j++){
                        diag[j] = max( diag[j], wa2[j] );
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        /** The inner loop. **/
        inner_done_first = false;
        do {
            if(get_local_id(0) == 0){
                /** Determine the Levenberg-Marquardt parameter. **/
                lm_lmpar(%(NMR_PARAMS)s, fjac, %(NMR_OBSERVATIONS)s, Pivot, diag, qtf, delta, lmpar, wa1, wa2, wf, wa3);
                /* used return values are fjac (partly), lmpar, wa1=x, wa3=diag*x */

                /* Predict scaled reduction */
                *pnorm = lm_euclidian_norm(wa3, %(NMR_PARAMS)s);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

			if(!isfinite(*pnorm)) {
				return 10;
			}

			if(get_local_id(0) == 0){
                *temp2 = *lmpar * ((*pnorm / *fnorm)*(*pnorm / *fnorm));
                for (j = 0; j < %(NMR_PARAMS)s; j++) {
                    wa3[j] = 0;
                    for (i = 0; i <= j; i++){
                        wa3[i] -= fjac[j*%(NMR_OBSERVATIONS)s+i] * wa1[Pivot[j]];
                    }
                }
                *temp1 = lm_euclidian_norm(wa3, %(NMR_PARAMS)s) / *fnorm;
                *temp1 *= *temp1;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if (!isfinite(*temp1)){
                return 10;
            }


            prered = *temp1 + 2 * *temp2;
            dirder = -*temp1 + *temp2; /* scaled directional derivative */

            /* At first call, adjust the initial step bound. */
            if (!outer_done_first && !inner_done_first && *pnorm < delta ){
                delta = *pnorm;
            }

            if(get_local_id(0) == 0){
                /** Evaluate the function at x + p. **/
                for (j = 0; j < %(NMR_PARAMS)s; j++){
                    wa2[j] = model_parameters[j] - wa1[j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            %(FUNCTION_NAME)s(wa2, data, wf);
            ++nfev;

            if(get_local_id(0) == 0){
                *fnorm1 = lm_euclidian_norm(wf, %(NMR_OBSERVATIONS)s);
                // exceptionally, for this norm we do not test for infinity
                // because we can deal with it without terminating.

			    /** Evaluate the scaled reduction. **/
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            /* actual scaled reduction (supports even the case fnorm1=infty) */
            if (0.1 * *fnorm1 < *fnorm)
                actred = 1 - ((*fnorm1 / *fnorm) * (*fnorm1 / *fnorm));
            else
                actred = -1;

            /* Ratio of actual to predicted reduction */
            ratio = 0;
            if(prered){
                ratio = actred / prered;
            }


            /* Update the step bound */
            if (ratio <= 0.25) {
                if (actred >= 0)
                    tmp = 0.5;
                else
                    tmp = 0.5 * dirder / (dirder + 0.5 * actred);
                if (0.1 * *fnorm1 >= *fnorm || tmp < 0.1)
                    tmp = 0.1;
                delta = tmp * min(delta, (mot_float_type)(*pnorm / 0.1));

                if(get_local_id(0) == 0){
                    *lmpar /= tmp;
                }
            } else if (*lmpar == 0 || ratio >= 0.75) {
                delta = 2 * *pnorm;

                if(get_local_id(0) == 0){
                    *lmpar *= 0.5;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            /**  On success, update solution, and test for convergence. **/
			inner_success = ratio >= 1e-4;

            if ( inner_success ) {

				if(get_local_id(0) == 0){
                    /* Update x, fvec, and their norms */
                    if (SCALE_DIAG) {
                        for (j = 0; j < %(NMR_PARAMS)s; j++) {
                            model_parameters[j] = wa2[j];
                            wa2[j] = diag[j] * model_parameters[j];
                        }
                    } else {
                        for (j = 0; j < %(NMR_PARAMS)s; j++){
                            model_parameters[j] = wa2[j];
                        }
                    }
                    for (i = 0; i < %(NMR_OBSERVATIONS)s; i++){
                        fvec[i] = wf[i];
                    }
                    *xnorm = lm_euclidian_norm(wa2, %(NMR_PARAMS)s);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (!isfinite(*xnorm)){
                    return 10; /* nan */
                }

                if(get_local_id(0) == 0){
                    *fnorm = *fnorm1;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            /* convergence tests */
            if (*fnorm <= LM_DWARF){
                return 1; /* success: sum of squares almost zero */
            }
            /* test two criteria (both may be fulfilled) */
            if (fabs(actred) <= FTOL && prered <= FTOL && ratio <= 2){
				if (delta <= XTOL * *xnorm){
				    return 4; /* success: sum of squares almost stable */
				}
				return 2; /* success: x almost stable */
			}

			/** Tests for termination and stringent tolerances. **/
			if ( nfev >= MAXFEV ){
                return 6;
            }
            if ( fabs(actred) <= LM_MACHEP && prered <= LM_MACHEP && ratio <= 2 ){
                return 7;
            }
            if ( delta <= LM_MACHEP * *xnorm ){
                return 8;
            }
            if ( *gnorm <= LM_MACHEP){
                return 9;
            }

			/** End of the inner loop. Repeat if iteration unsuccessful. **/
            inner_done_first = true;
        } while ( !inner_success );
        outer_done_first = true;
    };/***  End of the loop. ***/
} /*** lmmin. ***/


/*****************************************************************************/
/*  lm_lmpar (determine Levenberg-Marquardt parameter)                       */
/*****************************************************************************/

void lm_lmpar(
    const int n,
    local mot_float_type* const r,
    int ldr,
    local const int* const Pivot,
    local mot_float_type* const diag,
    local mot_float_type* const qtb,
    const mot_float_type delta,
    local mot_float_type * const par,
    local mot_float_type * const x,
    local mot_float_type * const Sdiag,
    local mot_float_type * const aux,
    local mot_float_type * const xdi)
{
/*     Given an m by n matrix A, an n by n nonsingular diagonal matrix D,
 *     an m-vector b, and a positive number delta, the problem is to
 *     determine a parameter value par such that if x solves the system
 *
 *          A*x = b  and  sqrt(par)*D*x = 0
 *
 *     in the least squares sense, and dxnorm is the Euclidean norm of D*x,
 *     then either par=0 and (dxnorm-delta) < 0.1*delta, or par>0 and
 *     abs(dxnorm-delta) < 0.1*delta.
 *
 *     Using lm_qrsolv, this subroutine completes the solution of the
 *     problem if it is provided with the necessary information from the
 *     QR factorization, with column pivoting, of A. That is, if A*P = Q*R,
 *     where P is a permutation matrix, Q has orthogonal columns, and R is
 *     an upper triangular matrix with diagonal elements of nonincreasing
 *     magnitude, then lmpar expects the full upper triangle of R, the
 *     permutation matrix P, and the first n components of Q^T*b. On output
 *     lmpar also provides an upper triangular matrix S such that
 *
 *          P^T*(A^T*A + par*D*D)*P = S^T*S.
 *
 *     S is employed within lmpar and may be of separate interest.
 *
 *     Only a few iterations are generally needed for convergence of the
 *     algorithm. If, however, the limit of 10 iterations is reached, then
 *     the output par will contain the best value obtained so far.
 *
 *     Parameters:
 *
 *      n is a positive integer INPUT variable set to the order of r.
 *
 *      r is an n by n array. On INPUT the full upper triangle must contain
 *        the full upper triangle of the matrix R. On OUTPUT the full upper
 *        triangle is unaltered, and the strict lower triangle contains the
 *        strict upper triangle (transposed) of the upper triangular matrix S.
 *
 *      ldr is a positive integer INPUT variable not less than n which
 *        specifies the leading dimension of the array R.
 *
 *      Pivot is an integer INPUT array of length n which defines the
 *        permutation matrix P such that A*P = Q*R. Column j of P is column
 *        Pivot(j) of the identity matrix.
 *
 *      diag is an INPUT array of length n which must contain the diagonal
 *        elements of the matrix D.
 *
 *      qtb is an INPUT array of length n which must contain the first
 *        n elements of the vector Q^T*b.
 *
 *      delta is a positive INPUT variable which specifies an upper bound
 *        on the Euclidean norm of D*x.
 *
 *      par is a nonnegative variable. On INPUT par contains an initial
 *        estimate of the Levenberg-Marquardt parameter. On OUTPUT par
 *        contains the final estimate.
 *
 *      x is an OUTPUT array of length n which contains the least-squares
 *        solution of the system A*x = b, sqrt(par)*D*x = 0, for the output par.
 *
 *      Sdiag is an array of length n needed as workspace; on OUTPUT it
 *        contains the diagonal elements of the upper triangular matrix S.
 *
 *      aux is a multi-purpose work array of length n.
 *
 *      xdi is a work array of length n. On OUTPUT: diag[j] * x[j].
 *
 */
    // used as both iter and nsing
    int iter, nsing;
    int i, j;
    mot_float_type gnorm, parc;
    mot_float_type dxnorm, fp, fp_old, parl, paru;
    mot_float_type temp;
    mot_float_type p1 = 0.1;

    /*** Compute and store in x the Gauss-Newton direction. If the Jacobian
         is rank-deficient, obtain a least-squares solution. ***/

    nsing = n;
    for (j = 0; j < n; j++) {
        aux[j] = qtb[j];
        if (r[j * ldr + j] == 0 && nsing == n){
            nsing = j;
        }
        if (nsing < n){
            aux[j] = 0;
        }
    }
    for (j = nsing - 1; j >= 0; j--) {
        aux[j] = aux[j] / r[j + ldr * j];
        temp = aux[j];
        for (i = 0; i < j; i++){
            aux[i] -= r[j * ldr + i] * temp;
        }
    }

    for (j = 0; j < n; j++){
        x[Pivot[j]] = aux[j];
    }

    /*** Initialize the iteration counter, evaluate the function at the origin,
         and test for acceptance of the Gauss-Newton direction. ***/

    for (j = 0; j < n; j++){
        xdi[j] = diag[j] * x[j];
    }
    dxnorm = lm_euclidian_norm(xdi, n);
    fp = dxnorm - delta;
    if (fp <= p1 * delta) {
        *par = 0;
        return;
    }

    /*** If the Jacobian is not rank deficient, the Newton step provides a
         lower bound, parl, for the zero of the function. Otherwise set this
         bound to zero. ***/

    parl = 0;
    if (nsing >= n) {
        for (j = 0; j < n; j++){
            aux[j] = diag[Pivot[j]] * xdi[Pivot[j]] / dxnorm;
        }

        for (j = 0; j < n; j++) {
            temp = 0;
            for (i = 0; i < j; i++){
                temp += r[j*ldr+i] * aux[i];
            }
            aux[j] = (aux[j] - temp) / r[j+ldr*j];
        }
        temp = lm_euclidian_norm(aux, n);
        parl = fp / delta / temp / temp;
    }

    /*** Calculate an upper bound, paru, for the zero of the function. ***/

    for (j = 0; j < n; j++) {
        temp = 0;
        for (i = 0; i <= j; i++){
            temp += r[j*ldr+i] * qtb[i];
        }
        aux[j] = temp / diag[Pivot[j]];
    }

    gnorm = lm_euclidian_norm(aux, n);
    paru = gnorm / delta;
    if (paru == 0){
        paru = LM_DWARF / min(delta, p1);
    }

    /*** If the input par lies outside of the interval (parl,paru),
         set par to the closer endpoint. ***/

    *par = max(*par, parl);
    *par = min(*par, paru);
    if (*par == 0){
        *par = gnorm / dxnorm;
    }

    /*** Iterate. ***/
    for (iter=0; ; iter++) {

        /** Evaluate the function at the current value of par. **/

        if (*par == 0){
            *par = max((mot_float_type)LM_DWARF, (mot_float_type)(0.001 * paru));
        }
        temp = sqrt(*par);
        for (j = 0; j < n; j++){
            aux[j] = temp * diag[j];
        }

        lm_qrsolv( n, r, ldr, Pivot, aux, qtb, x, Sdiag, xdi );
        /* return values are r, x, Sdiag */

        for (j = 0; j < n; j++){
            xdi[j] = diag[j] * x[j]; /* used as output */
        }
        dxnorm = lm_euclidian_norm(xdi, n);
        fp_old = fp;
        fp = dxnorm - delta;

        /** If the function is small enough, accept the current value
            of par. Also test for the exceptional cases where parl
            is zero or the number of iterations has reached 10. **/
        if (fabs(fp) <= p1 * delta ||
            (parl == 0 && fp <= fp_old && fp_old < 0) || iter == 10){
            break; /* the only exit from the iteration. */
        }

        /** Compute the Newton correction. **/
        for (j = 0; j < n; j++){
            aux[j] = diag[Pivot[j]] * xdi[Pivot[j]] / dxnorm;
        }

        for (j = 0; j < n; j++) {
            aux[j] = aux[j] / Sdiag[j];
            for (i = j+1; i < n; i++){
                aux[i] -= r[j*ldr+i] * aux[j];
            }
        }
        temp = lm_euclidian_norm(aux, n);
        parc = fp / delta / temp / temp;

        /** Depending on the sign of the function, update parl or paru. **/
        if (fp > 0){
            parl = max(parl, *par);
        }
        else{ /* fp < 0 [the case fp==0 is precluded by the break condition] */
            paru = min(paru, *par);
        }

        /** Compute an improved estimate for par. **/
        *par = max(parl, *par + parc);

    }

} /*** lm_lmpar. ***/

/******************************************************************************/
/*  lm_qrfac (QR factorization, from lapack)                                  */
/******************************************************************************/

void lm_qrfac(const int m, const int n, local mot_float_type* const A, local int* const Pivot,
              local mot_float_type* const Rdiag, local mot_float_type* const Acnorm, local mot_float_type* const W)
{
/*
 *     This subroutine uses Householder transformations with column pivoting
 *     to compute a QR factorization of the m by n matrix A. That is, qrfac
 *     determines an orthogonal matrix Q, a permutation matrix P, and an
 *     upper trapezoidal matrix R with diagonal elements of nonincreasing
 *     magnitude, such that A*P = Q*R. The Householder transformation for
 *     column k, k = 1,2,...,n, is of the form
 *
 *          I - 2*w*wT/|w|^2
 *
 *     where w has zeroes in the first k-1 positions.
 *
 *     Parameters:
 *
 *      m is an INPUT parameter set to the number of rows of A.
 *
 *      n is an INPUT parameter set to the number of columns of A.
 *
 *      A is an m by n array. On INPUT, A contains the matrix for which the
 *        QR factorization is to be computed. On OUTPUT the strict upper
 *        trapezoidal part of A contains the strict upper trapezoidal part
 *        of R, and the lower trapezoidal part of A contains a factored form
 *        of Q (the non-trivial elements of the vectors w described above).
 *
 *      Pivot is an integer OUTPUT array of length n that describes the
 *        permutation matrix P. Column j of P is column Pivot(j) of the
 *        identity matrix.
 *
 *      Rdiag is an OUTPUT array of length n which contains the diagonal
 *        elements of R.
 *
 *      Acnorm is an OUTPUT array of length n which contains the norms of
 *        the corresponding columns of the input matrix A. If this information
 *        is not needed, then Acnorm can share storage with Rdiag.
 *
 *      W is a work array of length n.
 *
 */
    int i, j, k, kmax;
    mot_float_type ajnorm, temp;

    /** Compute initial column norms;
        initialize Pivot with identity permutation. ***/
    for (j = 0; j < n; j++) {
        W[j] = Rdiag[j] = Acnorm[j] = lm_euclidian_norm(&A[j*m], m);
        Pivot[j] = j;
    }

    /** Loop over columns of A. **/
    for (j = 0; j < n; j++) {

        /** Bring the column of largest norm into the pivot position. **/
        kmax = j;
        for (k = j+1; k < n; k++)
            if (Rdiag[k] > Rdiag[kmax])
                kmax = k;

        if (kmax != j) {
            /* Swap columns j and kmax. */
            k = Pivot[j];
            Pivot[j] = Pivot[kmax];
            Pivot[kmax] = k;
            for (i = 0; i < m; i++) {
                temp = A[j*m+i];
                A[j*m+i] = A[kmax*m+i];
                A[kmax*m+i] = temp;
            }
            /* Half-swap: Rdiag[j], W[j] won't be needed any further. */
            Rdiag[kmax] = Rdiag[j];
            W[kmax] = W[j];
        }


        /** Compute the Householder reflection vector w_j to reduce the
            j-th column of A to a multiple of the j-th unit vector. **/
        ajnorm = lm_euclidian_norm(&A[j*m+j], m-j);
        if (ajnorm == 0) {
            Rdiag[j] = 0;
        }
        else{
	    /* Let the partial column vector A[j][j:] contain w_j := e_j+-a_j/|a_j|,
	       where the sign +- is chosen to avoid cancellation in w_jj. */
            if (A[j*m+j] < 0){
                ajnorm = -ajnorm;
            }
            for (i = j; i < m; i++){
                A[j*m+i] /= ajnorm;
            }
            A[j*m+j] += 1;

            /** Apply the Householder transformation U_w := 1 - 2*w_j.w_j/|w_j|^2
                to the remaining columns, and update the norms. **/
            for (k = j + 1; k < n; k++){
                /* Compute scalar product w_j * a_j. */
                temp = 0;
                for (i = j; i < m; i++){
                    temp += A[j*m+i] * A[k*m+i];
                }

                /* Normalization is simplified by the coincidence |w_j|^2=2w_jj. */
                temp = temp / A[j*m+j];

                /* Carry out transform U_w_j * a_k. */
                for (i = j; i < m; i++){
                    A[k*m+i] -= temp * A[j*m+i];
                }

                /* No idea what happens here. */
                if (Rdiag[k] != 0) {
                    temp = A[m*k+j] / Rdiag[k];
                    if (fabs(temp) < 1) {
                        Rdiag[k] *= sqrt(1 - (temp*temp));
                        temp = Rdiag[k] / W[k];
                    } else {
                        temp = 0;
                    }

                    if(temp == 0 || 0.05 * (temp * temp) <= LM_MACHEP){
                        Rdiag[k] = lm_euclidian_norm(&A[m*k+j+1], m-j-1);
                        W[k] = Rdiag[k];
                    }
                }
            }

            Rdiag[j] = -ajnorm;
        }
    }
} /*** lm_qrfac. ***/


/*****************************************************************************/
/*  lm_qrsolv (linear least-squares)                                         */
/*****************************************************************************/

void lm_qrsolv(const int n,
               local mot_float_type* const r,
               const int ldr,
               local const int* const Pivot,
               local const mot_float_type* const diag,
               local const mot_float_type* const qtb,
	           local mot_float_type* const x,
	           local mot_float_type* const Sdiag,
               local mot_float_type* const W)
{
/*
 *     Given an m by n matrix A, an n by n diagonal matrix D, and an
 *     m-vector b, the problem is to determine an x which solves the
 *     system
 *
 *          A*x = b  and  D*x = 0
 *
 *     in the least squares sense.
 *
 *     This subroutine completes the solution of the problem if it is
 *     provided with the necessary information from the QR factorization,
 *     with column pivoting, of A. That is, if A*P = Q*R, where P is a
 *     permutation matrix, Q has orthogonal columns, and R is an upper
 *     triangular matrix with diagonal elements of nonincreasing magnitude,
 *     then qrsolv expects the full upper triangle of R, the permutation
 *     matrix P, and the first n components of Q^T*b. The system
 *     A*x = b, D*x = 0, is then equivalent to
 *
 *          R*z = Q^T*b,  P^T*D*P*z = 0,
 *
 *     where x = P*z. If this system does not have full rank, then a least
 *     squares solution is obtained. On output qrsolv also provides an upper
 *     triangular matrix S such that
 *
 *          P^T*(A^T*A + D*D)*P = S^T*S.
 *
 *     S is computed within qrsolv and may be of separate interest.
 *
 *     Parameters:
 *
 *      n is a positive integer INPUT variable set to the order of R.
 *
 *      r is an n by n array. On INPUT the full upper triangle must contain
 *        the full upper triangle of the matrix R. On OUTPUT the full upper
 *        triangle is unaltered, and the strict lower triangle contains the
 *        strict upper triangle (transposed) of the upper triangular matrix S.
 *
 *      ldr is a positive integer INPUT variable not less than n which
 *        specifies the leading dimension of the array R.
 *
 *      Pivot is an integer INPUT array of length n which defines the
 *        permutation matrix P such that A*P = Q*R. Column j of P is column
 *        Pivot(j) of the identity matrix.
 *
 *      diag is an INPUT array of length n which must contain the diagonal
 *        elements of the matrix D.
 *
 *      qtb is an INPUT array of length n which must contain the first
 *        n elements of the vector Q^T*b.
 *
 *      x is an OUTPUT array of length n which contains the least-squares
 *        solution of the system A*x = b, D*x = 0.
 *
 *      Sdiag is an OUTPUT array of length n which contains the diagonal
 *        elements of the upper triangular matrix S.
 *
 *      W is a work array of length n.
 *
 */
    int i, kk, j, k, nsing;
    mot_float_type qtbpj, temp;
    mot_float_type _sin, _cos, _tan, _cot; /* local variables, not functions */

    /*** Copy R and Q^T*b to preserve input and initialize S.
         In particular, save the diagonal elements of R in x. ***/

    for (j = 0; j < n; j++) {
        for (i = j; i < n; i++)
            r[j*ldr+i] = r[i*ldr+j];
        x[j] = r[j*ldr+j];
        W[j] = qtb[j];
    }

    /*** Eliminate the diagonal matrix D using a Givens rotation. ***/

    for (j = 0; j < n; j++) {

        /*** Prepare the row of D to be eliminated, locating the diagonal
             element using P from the QR factorization. ***/

        if (diag[Pivot[j]] != 0) {
            for (k = j; k < n; k++)
                Sdiag[k] = 0;
            Sdiag[j] = diag[Pivot[j]];

            /*** The transformations to eliminate the row of D modify only
                 a single element of Q^T*b beyond the first n, which is
                 initially 0. ***/

            qtbpj = 0;
            for (k = j; k < n; k++) {

                /** Determine a Givens rotation which eliminates the
                    appropriate element in the current row of D. **/
                if (Sdiag[k] == 0){
                }
                else{
                    kk = k + ldr * k;
                    if (fabs(r[kk]) < fabs(Sdiag[k])) {
                        _cot = r[kk] / Sdiag[k];
                        _sin = 1 / hypot(1, _cot);
                        _cos = _sin * _cot;
                    } else {
                        _tan = Sdiag[k] / r[kk];
                        _cos = 1 / hypot(1, _tan);
                        _sin = _cos * _tan;
                    }

                    /** Compute the modified diagonal element of R and
                    	the modified element of (Q^T*b,0). **/
                    r[kk] = _cos * r[kk] + _sin * Sdiag[k];
                    temp = _cos * W[k] + _sin * qtbpj;
                    qtbpj = -_sin * W[k] + _cos * qtbpj;
                    W[k] = temp;

                    /** Accumulate the transformation in the row of S. **/
                    for (i = k+1; i < n; i++) {
                        temp = _cos * r[k * ldr + i] + _sin * Sdiag[i];
                        Sdiag[i] = -_sin * r[k * ldr + i] + _cos * Sdiag[i];
                        r[k * ldr + i] = temp;
                    }
                }
            }
        }
        /** Store the diagonal element of S and restore
            the corresponding diagonal element of R. **/

        Sdiag[j] = r[j * ldr + j];
        r[j * ldr + j] = x[j];
    }

    /*** Solve the triangular system for z. If the system is singular, then
        obtain a least-squares solution. ***/
    nsing = n;
    for (j = 0; j < n; j++) {
        if (Sdiag[j] == 0 && nsing == n){
            nsing = j;
        }
        if (nsing < n){
            W[j] = 0;
        }
    }

    for (j = nsing - 1; j >= 0; j--) {
        temp = 0;
        for (i = j + 1; i < nsing; i++){
            temp += r[j * ldr + i] * W[i];
        }
        W[j] = (W[j] - temp) / Sdiag[j];
    }

    /*** Permute the components of z back to components of x. ***/

    for (j = 0; j < n; j++)
        x[Pivot[j]] = W[j];

} /*** lm_qrsolv. ***/

/******************************************************************************/
/*  lm_enorm (Euclidean norm)                                                 */
/******************************************************************************/
double lm_euclidian_norm(local const mot_float_type* const x, const int n){
/*     This function calculates the Euclidean norm of an n-vector x.
 *
 *     The Euclidean norm is computed by accumulating the sum of squares
 *     in three different sums. The sums of squares for the small and large
 *     components are scaled so that no overflows occur. Non-destructive
 *     underflows are permitted. Underflows and overflows do not occur in
 *     the computation of the unscaled sum of squares for the intermediate
 *     components. The definitions of small, intermediate and large components
 *     depend on two constants, LM_SQRT_DWARF and LM_SQRT_GIANT. The main
 *     restrictions on these constants are that LM_SQRT_DWARF**2 not underflow
 *     and LM_SQRT_GIANT**2 not overflow.
 *
 *     Parameters:
 *
 *      n is a positive integer INPUT variable.
 *
 *      x is an INPUT array of length n.
 */
    int i;
    double agiant, s1, s2, s3, xabs, x1max, x3max;

    s1 = 0;
    s2 = 0;
    s3 = 0;
    x1max = 0;
    x3max = 0;
    agiant = LM_ENORM_SQRT_GIANT / n;

    /** Sum squares. **/
    for (i = 0; i < n; i++) {
        xabs = fabs(x[i]);
        if (xabs > LM_ENORM_SQRT_DWARF) {
            if (xabs < agiant) {
                s2 += xabs * xabs;
            } else if (xabs > x1max) {
                s1 = 1 + s1 * ((x1max / xabs) * (x1max / xabs));
                x1max = xabs;
            } else {
                s1 += ((xabs / x1max) * (xabs / x1max));
            }
        } else if (xabs > x3max) {
            s3 = 1 + s3 * ((x3max / xabs) * (x3max / xabs));
            x3max = xabs;
        } else if (xabs != 0) {
            s3 += ((xabs / x3max) * (xabs / x3max));
        }
    }

    /** Calculate the norm. **/
    if (s1 != 0)
        return x1max * sqrt(s1 + (s2 / x1max) / x1max);
    else if (s2 != 0)
        if (s2 >= x3max)
            return sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
        else
            return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
    else
        return x3max * sqrt(s3);


} /*** euclidian_norm. ***/


#undef LM_MACHEP
#undef LM_DWARF
#undef LM_SQRT_DWARF
#undef LM_SQRT_GIANT
#undef LM_USERTOL
#undef FTOL
#undef XTOL
#undef GTOL
#undef STEP_BOUND
#undef PATIENCE
#undef SCALE_DIAG
#undef MAXFEV
#undef LM_ENORM_SQRT_GIANT
#undef LM_ENORM_SQRT_DWARF

#endif // LMMIN_CL
