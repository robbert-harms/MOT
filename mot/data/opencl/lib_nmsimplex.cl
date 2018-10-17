#ifndef LIB_NMSIMPLEX_CL
#define LIB_NMSIMPLEX_CL

/**
 * Author = Robbert Harms
 * Date = 2014-09-29
 * License = see hereunder
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/*
 * Program: nmsimplex.c
 * Author : Michael F. Hutt, Robbert Harms
 * http://www.mikehutt.com
 * 11/3/97
 *
 * An implementation of the Nelder-Mead simplex method.
 *
 * Copyright (c) 1997-2011 <Michael F. Hutt>
 *
 * (Licence: X11 license)
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * Jan. 6, 1999
 * Modified to conform to the algorithm presented
 * in Margaret H. Wright's paper on Direct Search Methods.
 *
 * Jul. 23, 2007
 * Fixed memory leak.
 *
 * Mar. 1, 2011
 * Added constraints.
 *
 * 2014
 * Removed constraints since MOT features parameter transformations.
 */
#define USER_TOL_X  30*MOT_EPSILON              /** the precision we break at*/

/** The evaluation function we are expecting. */
double %(FUNCTION_NAME)s(local mot_float_type* x, void* data_void);

/*
 * Create the initial simplex.
 * This sets x_0 = x_input to allow for proper restarts and set the remaining vertices to the initial simplex scale.
 */
void _libnms_initialize_simplex(
        int nmr_parameters,
        local mot_float_type* vertices, // [n+1,n]
        local mot_float_type* const model_parameters, // [n]
        local mot_float_type* const initial_simplex_scale // [n]
){
    int i, j;

    for (i=0; i < nmr_parameters; i++) {
		vertices[0 * nmr_parameters + i] = model_parameters[i];
	}

	for (j=1; j <= nmr_parameters; j++) {
		for (i=0; i < nmr_parameters; i++) {
            /* There are two ways to create the initial simplex. The first is by:
                    x_j = x_input + h_j * e_j
               i.e. making the simplex right-angled at x0 based on coordinate axes,
                    where hj is a stepsize in the direction of unit vector ej in Rn.

               The second way is by creating a regular simplex where all edges have the same length.

               This implementation applies a variation of the above by creating a semi-regular simplex based
               on the initial scale for each parameter:
                    x_j = h_j * e_j  for i == (j-1)
                    x_j = x_input    for i != (j-1)
            */
            if(i == (j-1)){
                vertices[j * nmr_parameters + i] = initial_simplex_scale[i];
            }
            else{
                vertices[j * nmr_parameters + i] = model_parameters[i];
            }
		}
	}
}

/* find the initial function values */
void _libnms_initialize_function_values(
        int nmr_parameters,
        local mot_float_type* vertices, // [n+1,n],
        local mot_float_type* func_vals, // [n+1]
        void* data){

	for (int j=0; j < nmr_parameters + 1; j++) {
		func_vals[j] = %(FUNCTION_NAME)s(vertices + j * nmr_parameters, data);
	}
}


/* find the index of the largest and smallest value */
void _libnms_find_worst_best_fvals(
        int nmr_parameters,
        local mot_float_type* func_vals, // [n+1]
        int* ind_worst, int* ind_best){
    int i;

    *ind_worst=0;
    *ind_best=0;

    for (i=0; i<=nmr_parameters; i++) {
        /* find largest */
        if (func_vals[i] > func_vals[*ind_worst]) {
            *ind_worst = i;
        }

        /* find smallest */
        if (func_vals[i] < func_vals[*ind_best]) {
            *ind_best = i;
        }
    }
}

/*
 * Determine the indices of the worst, second worst and the best vertex in the current working simplex S
 */
void _libnms_find_ordering_indices(int nmr_parameters,
                                   local mot_float_type* func_vals, // [n+1]
                                   int* ind_worst, int* ind_best,
                                   int* ind_second_worst){

    int i;

    _libnms_find_worst_best_fvals(nmr_parameters, func_vals, ind_worst, ind_best);

    /* find the index of the second largest value */
    *ind_second_worst=*ind_best;

    for (i=0; i<=nmr_parameters; i++) {
        if (func_vals[i] > func_vals[*ind_second_worst] && func_vals[i] < func_vals[*ind_worst]) {
            *ind_second_worst = i;
        }
    }
}

/**
 * Calculate the variance of the given input values
 */
mot_float_type _libnms_get_variance(local mot_float_type* values, int n){
    /** Online variance algorithm by Welford
     *  B. P. Welford (1962)."Note on a method for calculating corrected sums of squares
     *      and products". Technometrics 4(3):419-420.
     *
     * Also studied in:
     * Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1983).
     *      Algorithms for Computing the Sample Variance: Analysis and Recommendations.
     *      The American Statistician 37, 242-247. http://www.jstor.org/stable/2683386
     */
    mot_float_type mean, M2, delta;
    int j;

    mean = 0;
    M2 = 0;

    for(j = 0; j < n; j++){
        delta = values[j] - mean;
        mean += delta / (j + 1);
        M2 += delta * (values[j] - mean);
    }

    return M2 / (n - 1);
}

/**
 * Calculate the centroid c of the best side, this is the one opposite to the worst vertex.
 *
 * Puts the result in the centroid array.
 */
void _libnms_calculate_centroid(
        int nmr_parameters,
        local mot_float_type* vertices, // [n+1,n],
        local mot_float_type* centroid, // [n]
        int ind_worst){

    if(get_local_id(0) == 0){
        int i, j;
        double tmp;

        for (j=0; j < nmr_parameters; j++) {

            tmp=0.0;
            for (i=0; i <= nmr_parameters; i++) {
                if (i!=ind_worst) {
                    tmp += vertices[i * nmr_parameters + j];
                }
            }

            centroid[j] = tmp/nmr_parameters;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/**
 * Calculate the reflection and accept it if better.
 *
 * At output the following arrays are changed:
 *    vertices: the worst point is possibly updated with the reflected point
 *    tmp_vertex: the reflection point
 *    func_vals: the worst point is possibly updated with the reflected function value
 *    reflection_fval: set to the function value of the reflection point
 *
 * Returns:
 *     true if the new point was accepted, false otherwise
 */
bool _libnms_simplex_reflect(
        int nmr_parameters,
        local mot_float_type* const vertices, // [n+1,n]
        local const mot_float_type* const centroid, // [n]
        local mot_float_type* const tmp_vertex, // [n]
        local mot_float_type* const func_vals, // [n+1]
        mot_float_type* const reflection_fval, // [1]
        const int ind_best,
        const int ind_second_worst,
        const int ind_worst,
        const mot_float_type alpha,
        void* data){

    int j;

    if(get_local_id(0) == 0){
        for (j=0; j< nmr_parameters;j++) {
            tmp_vertex[j] = centroid[j] + alpha * (centroid[j] - vertices[ind_worst * nmr_parameters + j]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    *reflection_fval = %(FUNCTION_NAME)s(tmp_vertex, data);

    if(*reflection_fval < func_vals[ind_second_worst] && *reflection_fval >= func_vals[ind_best]){
        if(get_local_id(0) == 0){
            for(j=0; j < nmr_parameters; j++){
                vertices[ind_worst * nmr_parameters + j] = tmp_vertex[j];
            }
            func_vals[ind_worst] = *reflection_fval;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        return 1;
    }
    return 0;
}

/**
 * Calculate the expansion and accept the correct new point
 *
 * At output the following arrays are changed:
 *    vertices: updated with the new point
 *    func_vals: the worst index is updated with the new point
 */
void _libnms_simplex_expand(
        int nmr_parameters,
        local mot_float_type* vertices, // [n+1,n]
        local const mot_float_type* const centroid, // [n]
        local const mot_float_type* const tmp_vertex, // [n]
        local mot_float_type* const func_vals, // [n+1]
        const mot_float_type reflection_fval,
        const int ind_best,
        const int ind_second_worst,
        const int ind_worst,
        const mot_float_type gamma,
        void* data){

    int j;
    mot_float_type expansion_fval;

    if(get_local_id(0) == 0){
        for (j=0; j < nmr_parameters; j++) {
            vertices[ind_worst * nmr_parameters + j] = centroid[j] + gamma * (tmp_vertex[j] - centroid[j]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    expansion_fval = %(FUNCTION_NAME)s(vertices + ind_worst * nmr_parameters, data);

    if (expansion_fval < reflection_fval){
        if(get_local_id(0) == 0){
            func_vals[ind_worst] = expansion_fval;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    else {
        if(get_local_id(0) == 0){
            for (j=0; j < nmr_parameters; j++) {
                vertices[ind_worst * nmr_parameters + j] = tmp_vertex[j];
            }
            func_vals[ind_worst] = reflection_fval;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


/**
 * Calculate the contraction and accept it if better.
 *
 * At output the following arrays are changed:
 *    vertices: the worst point is possibly updated with the reflected point
 *    tmp_vertex: updated to now reflect the contraction point
 *    func_vals: the worst point is possibly updated with the reflected function value
 *
 * Returns:
 *     true if the new point was accepted, false otherwise
 */
bool _libnms_simplex_contract(
        int nmr_parameters,
        local mot_float_type* const vertices, // [n+1,n]
        local const mot_float_type* const centroid, // [n]
        local mot_float_type* const tmp_vertex, // [n]
        local mot_float_type* const func_vals, // [n+1]
        const mot_float_type reflection_fval,
        const int ind_best,
        const int ind_second_worst,
        const int ind_worst,
        const mot_float_type beta,
        void* data){

    int j;
    mot_float_type contraction_fval;

    if (reflection_fval < func_vals[ind_worst] && reflection_fval >= func_vals[ind_second_worst]){
        /* perform outside contraction */
        for (j=0; j< nmr_parameters; j++) {
            tmp_vertex[j] = centroid[j] + beta * (tmp_vertex[j] - centroid[j]);
        }
    }
    else {
        /* perform inside contraction */
        for (j=0;j < nmr_parameters;j++) {
            tmp_vertex[j] = centroid[j] - beta * (centroid[j] - vertices[ind_worst * nmr_parameters + j]);
        }
    }

    contraction_fval = %(FUNCTION_NAME)s(tmp_vertex, data);

    if (contraction_fval < func_vals[ind_worst]) {
        for (j=0; j < nmr_parameters; j++) {
            vertices[ind_worst * nmr_parameters + j] = tmp_vertex[j];
        }
        func_vals[ind_worst] = contraction_fval;
        return true;
    }
    return false;
}

/**
 * The library function for performing Nelder-Mead simplex optimization
 *
 *
 * Args:
 * - nmr_parameters: the number of parameters in the problem
 * - model_parameters: IN: the initial set of parameters, OUT: the optimal set of parameters
 * - data: the function evaluation data
 * - initial_simplex_scale: the step sizes for creating the initial simplex, of size [nmr_parameters]
 * - fdiff: this is set, on output, to contain the difference between the high and low function values of the last simplex.
 * - psi: alternative stopping criteria used in the Sbplex method. if psi > 0, then it *replaces* the xtol and ftol
 *        stopping criteria with that the simplex diameter |xl - xh| must be reduced by a factor of psi
 *        this is for when nmsimplex is used within the subplex method; for
 *        ordinary termination tests, set psi = 0.
 * - max_iterations: the maximum number of iterations the simplex can run
 * - alpha, beta, gamma, delta: simplex strategy.
 * - scratch: the scratch array containing the memory we can use for the operations, of size [nmr_parameters * 3 + (nmr_parameters + 1)^2]
 */
int lib_nmsimplex(
        int nmr_parameters,
        local mot_float_type* const model_parameters,
        void* data,
        local mot_float_type* initial_simplex_scale,
        mot_float_type* fdiff,
        mot_float_type psi,
        int max_iterations,
        mot_float_type alpha,
        mot_float_type beta,
        mot_float_type gamma,
        mot_float_type delta,
        local mot_float_type* scratch // size: nmr_parameters * 3 + (nmr_parameters + 1)^2
        ){

    int return_code = 6;         /** the default return code is that we exhausted our patience */
	int ind_best;                /* vertex with smallest value */
	int ind_second_worst;        /* vertex with next largest value */
	int ind_worst;               /* vertex with largest value */
    int i, j;                   /** helper variables */
	int itr;	                /* track the number of iterations */
	double tmp;

	mot_float_type reflection_fval;      /* value of function at reflection point */

    local mot_float_type* contraction_tolerance = scratch; /* used for the subplex convergence check */
	local mot_float_type* centroid = contraction_tolerance + 1; /* centroid - coordinates, size [n] */
	local mot_float_type* tmp_vertex = centroid + nmr_parameters; /* simplex moving coordinates , size [n] */
    local mot_float_type* func_vals = tmp_vertex + nmr_parameters; /* value of function at each vertex, size [n + 1] */
    local mot_float_type* vertices = func_vals + (nmr_parameters + 1);   /* holds vertices of simplex, size [(n+1) * n] */

    *fdiff = HUGE_VAL;

    if(get_local_id(0) == 0){
	    _libnms_initialize_simplex(nmr_parameters, vertices, model_parameters, initial_simplex_scale);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

    _libnms_initialize_function_values(nmr_parameters, vertices, func_vals, data);

    if(psi > 0){
        /* For the psi convergence test we need to compute the size of the initial simplex for later comparison.
         * This diameter is calculated by the Euclidean distance between the largest and smallest vertices.
         */
        _libnms_find_worst_best_fvals(nmr_parameters, func_vals, &ind_worst, &ind_best);

        if(get_local_id(0) == 0){
            for(i = 0; i < nmr_parameters; ++i){
                *contraction_tolerance += pown(vertices[ind_best * nmr_parameters + i]
                                              - vertices[ind_worst * nmr_parameters + i], 2);
            }
            *contraction_tolerance = sqrt(*contraction_tolerance) * psi;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	/* begin the main loop of the minimization */
	for (itr=0; itr <= max_iterations; itr++) {
		_libnms_find_ordering_indices(nmr_parameters, func_vals, &ind_worst, &ind_best, &ind_second_worst);
        _libnms_calculate_centroid(nmr_parameters, vertices, centroid, ind_worst);

        /* use the default NMSimplex convergence criteria */
        if (sqrt(_libnms_get_variance(func_vals, nmr_parameters + 1)) < USER_TOL_X){
            return_code = 1;
            break;
        }

        /* additionally and optionally use the Subplex convergence criteria */
        if(psi > 0){
            tmp = 0;
            for(i = 0; i < nmr_parameters; ++i){
                tmp += pown(vertices[ind_best * nmr_parameters + i] - vertices[ind_worst * nmr_parameters + i], 2);
            }

            if(sqrt(tmp) < *contraction_tolerance){
                return_code = 11;
                break;
            }
        }

		/* reflect worst vertex to new vertex tmp_vertex */
		if(_libnms_simplex_reflect(nmr_parameters, vertices, centroid, tmp_vertex, func_vals, &reflection_fval, ind_best,
		                           ind_second_worst, ind_worst, alpha, data)){
            continue;
        }

		/* investigate a step further in this direction */
		if(reflection_fval < func_vals[ind_best]){
            _libnms_simplex_expand(nmr_parameters, vertices, centroid, tmp_vertex, func_vals, reflection_fval,
                                   ind_best, ind_second_worst, ind_worst, gamma, data);
            // we always accept one of the expansion point, so to the next iteration
            continue;
		}

		/* check to see if a contraction is necessary */
		if (reflection_fval >= func_vals[ind_second_worst]){
            if(_libnms_simplex_contract(nmr_parameters, vertices, centroid, tmp_vertex, func_vals, reflection_fval, ind_best,
                                        ind_second_worst, ind_worst, beta, data)){
                continue;
            }
		}

        /* If we get here none of the other operations were successful, so we apply the shrink operation.*/
        if(get_local_id(0) == 0){
            for (i=0; i < nmr_parameters + 1;i++) {
                if (i != ind_best) {
                    for (j=0; j < nmr_parameters; j++) {
                        vertices[i * nmr_parameters + j] =
                            vertices[ind_best * nmr_parameters + j]
                                + (vertices[i * nmr_parameters + j]-vertices[ind_best * nmr_parameters + j]) * delta;
                    }
                }
            }
        }
	    barrier(CLK_LOCAL_MEM_FENCE);

        for (i=0; i < nmr_parameters + 1;i++) {
            if (i != ind_best) {
                func_vals[i] = %(FUNCTION_NAME)s(vertices + i * nmr_parameters, data);
            }
        }
	}
	/* end main loop of the minimization */

	/* find the index of the smallest and largest value */
	_libnms_find_worst_best_fvals(nmr_parameters, func_vals, &ind_worst, &ind_best);

    /** set the results */
    if(get_local_id(0) == 0){
	    for (j=0;j<nmr_parameters;j++) {
            model_parameters[j] = vertices[ind_best * nmr_parameters + j];
        }
	}
	barrier(CLK_LOCAL_MEM_FENCE);

    /* set fdiff to the difference between the largest and smallest vertex */
    *fdiff = func_vals[ind_worst] - func_vals[ind_best];

	return return_code;
}

#undef USER_TOL_X

#endif // LIB_NMSIMPLEX_CL
