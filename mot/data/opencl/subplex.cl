#ifndef SUBPLEX_CL
#define SUBPLEX_CL

/**
 * Author = Robbert Harms
 * Date = 2017-05-23
 * License = L-GPLv3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/** This requires the following call arguments in addition to the NMR_PARAMS attribute */
#define ALPHA     %(ALPHA)r                   /* reflection coefficient, default 1 */
#define BETA      %(BETA)r                    /* contraction coefficient, default 0.5 */
#define GAMMA     %(GAMMA)r                   /* expansion coefficient default 2 */
#define DELTA     %(DELTA)r                   /* reduction coefficient default 0.5 */
#define PSI       %(PSI)r                     /* simplex reduction coefficient, default 0.25 */
#define OMEGA     %(OMEGA)r                   /* step reduction coefficient, default 0.1 */

/** If we use adaptive scales for ALPHA, BETA, GAMMA and DELTA in the NMSimplex calls. */
#define ADAPTIVE_SCALES %(ADAPTIVE_SCALES)d

/* the minimum subspace dimension, defaults to min(2, n) */
#define MIN_SUBSPACE_LENGTH %(MIN_SUBSPACE_LENGTH)r

/* the maximum subspace dimension, defaults to min(5, n) */
#define MAX_SUBSPACE_LENGTH %(MAX_SUBSPACE_LENGTH)r
///** This should hold for the min subspace dim and the maximum subspace dim: (1 <= nsmin <= nsmax <= n and nsmin*ceil(n/nsmax) <= n) */

#define MAX_IT    (%(PATIENCE)r * (%(NMR_PARAMS)r+1))
/** the precision we break at*/
#define USER_TOL_X  30*MOT_EPSILON

/** The evaluation function we are expecting. */
double %(FUNCTION_NAME)s(local mot_float_type* x, void* data_void);

// We define the header now and import the body later after having defined the subspace evaluation function
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
        local mot_float_type* scratch);


// the data wrapper used for the subspace evaluation function
typedef struct {
     local int *x_indices; /* subspace index permutation */
     int subspace_starting_index; /* starting index for this subspace */
     int subspace_length; /* dimension of subspace */
     local mot_float_type *x; /* current x vector */
     void* data; /* the "actual" underlying function data */
} SubspaceData;


/**
 * The evaluation function used by the NMSimplex calls
 *
 * The NMSimplex routine only optimizes subsets of the parameters at a time with all other parameters held constant.
 *
 */

double subspace_evaluate(local mot_float_type* subspace_model_parameters, void* subspace_data){

    SubspaceData* d = (SubspaceData*) subspace_data;
    local mot_float_type* x = d->x;

    if(get_local_id(0) == 0){
        for(int i = d->subspace_starting_index; i < d->subspace_starting_index + d->subspace_length; i++){
            x[(d->x_indices)[i]] = subspace_model_parameters[i - d->subspace_starting_index];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return %(FUNCTION_NAME)s(x, d->data);
}


/**
 * Sort the indices such that the absolute values are in increasing order.
 *
 * Before sorting the indices are reset to range(n).
 */
void _subplex_sort_indices(local const mot_float_type* const values, local int* indices, int n) {
    int h, i, j, tmp_ind;
    mot_float_type tmp_val;

    for(i = 0; i < n; ++i){
        indices[i] = i;
    }

    for(h = n; h /= 2;){
        for (i = h; i < n; i++) {
            tmp_ind = indices[i];
            tmp_val = fabs(values[indices[i]]);

            for (j = i; j >= h && tmp_val >= fabs(values[indices[j - h]]); j -= h) {
                indices[j] = indices[j - h];
            }

            indices[j] = tmp_ind;
        }
    }
}

/*
 * Calculate the L1 norm of a subset of the given array, using the given indices for indexing the values.
 *
 * This is a helper routine for the function _find_next_subspace_length()
 *
 * This computes the norm over [start, end)
 *
 * Returns:
 *  the l1norm
 */
mot_float_type _subplex_l1norm_subset(local const mot_float_type* const values,
                                     local const int* const indices, int start, int end){
    mot_float_type l1norm = 0;
    for(int i = start; i < end; i++){
        l1norm += fabs(values[indices[i]]);
    }
    return l1norm / (end - start);
}

/**
 * Find the next subspace dimension starting the search from the given starting index
 *
 * Args:
 *  - delta_x: the progress vector
 *  - x_indices: indices to the delta_x in sorted decreasing order
 *  - nmr_parameters: the length of the delta_x vector
 *  - min_subspace_length: the minimum subspace dimension
 *  - max_subspace_length: the maximum subspace dimension
 *  - starting_index: the index from which point on to search for the next dimension
 *
 * Returns:
 *  the size of the next dimension, counted from the given starting index
 */
int _subplex_find_next_subspace_length(local const mot_float_type* const delta_x,
                                      local const int* const x_indices,
                                      int nmr_parameters, int min_subspace_length, int max_subspace_length,
                                      int starting_index){

    int best_dimension_size = 0;
    mot_float_type best_fval = -HUGE_VAL;
    int dimension_size;
    mot_float_type current_fval;

    int remaining_length = nmr_parameters - starting_index;

    if((min_subspace_length * ceil((float)(remaining_length- min_subspace_length) / max_subspace_length)) > (remaining_length - min_subspace_length)){
        return remaining_length;
    }

    for(dimension_size = min_subspace_length;
        dimension_size <= max_subspace_length && min_subspace_length * ceil((float)(remaining_length - dimension_size)
                                                                            / max_subspace_length) <= (remaining_length - dimension_size);
        dimension_size++){

        current_fval = _subplex_l1norm_subset(delta_x, x_indices, starting_index, dimension_size + starting_index);

        if(dimension_size < remaining_length){
            current_fval -= _subplex_l1norm_subset(delta_x, x_indices, dimension_size + starting_index, remaining_length + starting_index);
        }

        if(current_fval > best_fval){
            best_dimension_size = dimension_size;
            best_fval = current_fval;
        }
    }

    return best_dimension_size;
}

/**
 * Get the subspaces we will use in the NMSimplex search
 *
 * Args:
 *  - delta_x: progress vector of the parameters
 *  - x_indices: set to contain the linear indexing of the subspaces
 *  - subspace_dimensions: OUT: list of subspace dimensions, this must at least be of length [floor(n/min_subspace_length)]
 *  - nmr_subspaces: OUT: the number of subspaces
 *  - nmr_parameters: the number of parameters
 *  - min_subspace_length: the minimum subspace size
 *  - max_subspace_length: the maximum subspace size
 */
void _subplex_get_subspaces(local const mot_float_type* const delta_x,
                            local int* x_indices,
                            local int* subspace_dimensions,
                            local int* nmr_subspaces,
                            int nmr_parameters,
                            int min_subspace_length,
                            int max_subspace_length){

    _subplex_sort_indices(delta_x, x_indices, nmr_parameters);

    int total_subspace_length = 0;
    int next_length;
    *nmr_subspaces = 0;

    while(total_subspace_length != nmr_parameters){
        next_length = _subplex_find_next_subspace_length(delta_x, x_indices, nmr_parameters, min_subspace_length,
                                                        max_subspace_length, total_subspace_length);
        total_subspace_length += next_length;
        subspace_dimensions[*nmr_subspaces] = next_length;
        (*nmr_subspaces)++;
    }
}


int subplex_minimize(local mot_float_type* model_parameters, /* in: initial guess, out: minimizer */
			       void* data,
			       local const mot_float_type* const xstep0,/* initial step sizes */
			       local mot_float_type* subplex_scratch_float,
			       local int* subplex_scratch_int){

    local mot_float_type* scratch_ind_float = subplex_scratch_float;

    local mot_float_type* fdiff_max = scratch_ind_float;                    scratch_ind_float += 1;
    local mot_float_type* step_size_scale = scratch_ind_float;              scratch_ind_float += 1;
    local mot_float_type* stepnorm = scratch_ind_float;                     scratch_ind_float += 1;
    local mot_float_type* dxnorm = scratch_ind_float;                       scratch_ind_float += 1;

    local mot_float_type* xstep = scratch_ind_float;                        scratch_ind_float += %(NMR_PARAMS)r;
    local mot_float_type* delta_x = scratch_ind_float;                      scratch_ind_float += %(NMR_PARAMS)r;
    local mot_float_type* subspace_model_parameters = scratch_ind_float;    scratch_ind_float += MAX_SUBSPACE_LENGTH;
    local mot_float_type* subspace_xstep = scratch_ind_float;               scratch_ind_float += MAX_SUBSPACE_LENGTH;
    local mot_float_type* nms_scratch = scratch_ind_float;

    local int* scratch_ind_int = subplex_scratch_int;

    local int* nmr_subspaces = scratch_ind_int;             scratch_ind_int += 1;
    local int* subspace_starting_index = scratch_ind_int;   scratch_ind_int += 1;
    /* permuted indices of model_parameters sorted by decreasing magnitude |delta_x| */
    local int* x_indices = scratch_ind_int;                 scratch_ind_int += %(NMR_PARAMS)r;
    local int* subspace_dimensions = scratch_ind_int;

    int i, k;
    int itr;
    SubspaceData subspace_data;
    mot_float_type fdiff;

    mot_float_type alpha = ALPHA;
    mot_float_type beta = BETA;
    mot_float_type gamma = GAMMA;
    mot_float_type delta = DELTA;

    if(get_local_id(0) == 0){
        for(i = 0; i < %(NMR_PARAMS)r; i++){
            xstep[i] = xstep0[i];
            delta_x[i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    subspace_data.x_indices = x_indices;
    subspace_data.x = model_parameters;
    subspace_data.data = data;

    for(itr=0; itr < MAX_IT; itr++) {

        if(get_local_id(0) == 0){
            // first use delta_x to create the subspaces
            _subplex_get_subspaces(delta_x, x_indices, subspace_dimensions, nmr_subspaces, %(NMR_PARAMS)r,
                                  MIN_SUBSPACE_LENGTH, MAX_SUBSPACE_LENGTH);


            // then use delta_x as a temporary container for the current parameters
            for(i = 0; i < %(NMR_PARAMS)r; i++){
                delta_x[i] = model_parameters[i];
            }

            *subspace_starting_index = 0; // loop variable, keeping track of the subspace index
            *fdiff_max = 0; // records the largest gain in function value over the subspaces
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // run NMSimplex on each subspace
        for(i = 0; i < *nmr_subspaces; i++){

            // prepare the subspace data
            subspace_data.subspace_starting_index = *subspace_starting_index;
            subspace_data.subspace_length = subspace_dimensions[i];

            if(get_local_id(0) == 0){
                // prepare the subspace parameters and step sizes (initial simplex scales)
                for(k = *subspace_starting_index; k < subspace_dimensions[i] + *subspace_starting_index; k++){
                    subspace_model_parameters[k - *subspace_starting_index] = model_parameters[x_indices[k]];
                    subspace_xstep[k - *subspace_starting_index] = xstep[x_indices[k]];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(ADAPTIVE_SCALES){
                alpha = 1;
                beta = 0.75 - 1.0 / (2 * subspace_dimensions[i]);
                gamma = 1 + 2.0 / subspace_dimensions[i];
                delta = 1 - 1.0 / subspace_dimensions[i];
            }

            lib_nmsimplex(subspace_dimensions[i], subspace_model_parameters, (void*)&subspace_data, subspace_xstep,
                          &fdiff, PSI, %(PATIENCE_NMSIMPLEX)r * (subspace_dimensions[i] + 1),
                          alpha, beta, gamma, delta, nms_scratch);

            if(get_local_id(0) == 0){
                // add the optimized subspace parameters to the current optimal set of model_parameters
                for(k = *subspace_starting_index; k < subspace_dimensions[i] + *subspace_starting_index; k++){
                    model_parameters[x_indices[k]] = subspace_model_parameters[k - *subspace_starting_index];
                }

                if(fdiff > *fdiff_max){
                    *fdiff_max = fdiff;
                }

                // prepare for the next iteration
                *subspace_starting_index += subspace_dimensions[i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(get_local_id(0) == 0){
            // compute change in optimal point, the previous delta_x contained the previous set of model parameters
            for (i = 0; i < %(NMR_PARAMS)r; ++i){
                delta_x[i] = model_parameters[i] - delta_x[i];
            }

            *dxnorm = 0;
            *stepnorm = 0;
            for (i = 0; i < %(NMR_PARAMS)r; ++i) {
                *dxnorm = max(*dxnorm, fabs(delta_x[i]));
                *stepnorm = max(*stepnorm, fabs(xstep[i]));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // stopping criteria using the infinity norm
        if(max(*dxnorm, (mot_float_type)(*stepnorm * PSI)) / max(*stepnorm, (mot_float_type)1.0) <= USER_TOL_X){
            return 3;
        }

        /**************************/
        // calculate the step size
        /**************************/
        if(get_local_id(0) == 0){
            if(*nmr_subspaces == 1){
                *step_size_scale = PSI;
            }
            else {
                // calculate the L1 norm
                *dxnorm = 0;
                *stepnorm = 0;
                for (i = 0; i < %(NMR_PARAMS)r; ++i) {
                    *dxnorm += fabs(delta_x[i]);
                    *stepnorm += fabs(xstep[i]);
                }
                *step_size_scale = min(max(*dxnorm / *stepnorm, (mot_float_type)OMEGA), (mot_float_type)(1.0/OMEGA));
            }

            // create the new step size (initial simplex scale)
            for (i = 0; i < %(NMR_PARAMS)r; ++i){
                if(delta_x[i] == 0){
                    xstep[i] = -(xstep[i] * *step_size_scale);
                }
                else{
                    xstep[i] = copysign(xstep[i] * *step_size_scale, delta_x[i]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /**************************/
        // calculate the step size
        /**************************/
    }

    return 6;
}

int subplex(local mot_float_type* const model_parameters, void* data,
            local mot_float_type* initial_simplex_scale,
            local mot_float_type* subplex_scratch_float,
            local int* subplex_scratch_int){

    if(get_local_id(0) == 0){
        %(INITIAL_SIMPLEX_SCALES)s
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return subplex_minimize(model_parameters, data, initial_simplex_scale, subplex_scratch_float, subplex_scratch_int);
}


#endif // SUBPLEX_CL
