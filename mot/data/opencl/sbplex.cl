#ifndef SBPLEX_CL
#define SBPLEX_CL

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
#define PSI       %(PSI)r                   /* simplex reduction coefficient, default 0.25 */
#define OMEGA     %(OMEGA)r                 /* step reduction coefficient, default 0.1 */

/* the minimum subspace dimension, defaults to min(2, n) */
#define MIN_SUBSPACE_LENGTH %(MIN_SUBSPACE_LENGTH)r

/* the maximum subspace dimension, defaults to min(5, n) */
#define MAX_SUBSPACE_LENGTH %(MAX_SUBSPACE_LENGTH)r
///** This should hold for the min subspace dim and the maximum subspace dim: (1 <= nsmin <= nsmax <= n and nsmin*ceil(n/nsmax) <= n) */

#define MAX_IT    (%(PATIENCE)r * (%(NMR_PARAMS)r+1))
/** the precision we break at*/
#define USER_TOL_X  30*MOT_EPSILON

// We define the header now and import the body later after having defined the subspace evaluation function
extern int lib_nmsimplex(
        int nmr_parameters,
        mot_float_type* const model_parameters,
        void* data,
        mot_float_type* initial_simplex_scale,
        mot_float_type* fdiff,
        mot_float_type psi,
        int max_iterations,
        mot_float_type alpha,
        mot_float_type beta,
        mot_float_type gamma,
        mot_float_type delta,
        mot_float_type* scratch);


// the data wrapper used for the subspace evaluation function
typedef struct {
     int *x_indices; /* subspace index permutation */
     int subspace_starting_index; /* starting index for this subspace */
     int subspace_length; /* dimension of subspace */
     mot_float_type *x; /* current x vector */
     void* data; /* the "actual" underlying function data */
} SubspaceData;


// the evaluation function used by the simplex calls
double subspace_evaluate(mot_float_type* subspace_model_parameters, void* subspace_data){

    SubspaceData* d = (SubspaceData*) subspace_data;
    mot_float_type* x = d->x;

    for(int i = d->subspace_starting_index; i < d->subspace_starting_index + d->subspace_length; i++){
        x[(d->x_indices)[i]] = subspace_model_parameters[i - d->subspace_starting_index];
    }

    return evaluate(x, d->data);
}


/**
 * Sort the indices such that the absolute values are in increasing order.
 *
 * Before sorting the indices are reset to range(n).
 */
void _sbplex_sort_indices(const mot_float_type* const values, int* indices, int n) {
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
mot_float_type _sbplex_l1norm_subset(const mot_float_type* const values, const int* const indices, int start, int end){
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
int _sbplex_find_next_subspace_length(const mot_float_type* const delta_x,
                                      const int* const x_indices,
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

        current_fval = _sbplex_l1norm_subset(delta_x, x_indices, starting_index, dimension_size + starting_index);

        if(dimension_size < remaining_length){
            current_fval -= _sbplex_l1norm_subset(delta_x, x_indices, dimension_size + starting_index, remaining_length + starting_index);
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
void _sbplex_get_subspaces(const mot_float_type* const delta_x,
                           int* x_indices, int* subspace_dimensions, int* nmr_subspaces,
                           int nmr_parameters, int min_subspace_length, int max_subspace_length){

    _sbplex_sort_indices(delta_x, x_indices, nmr_parameters);

    int total_subspace_length = 0;
    int next_length;
    *nmr_subspaces = 0;

    while(total_subspace_length != nmr_parameters){
        next_length = _sbplex_find_next_subspace_length(delta_x, x_indices, nmr_parameters, min_subspace_length,
                                                        max_subspace_length, total_subspace_length);
        total_subspace_length += next_length;
        subspace_dimensions[*nmr_subspaces] = next_length;
        (*nmr_subspaces)++;
    }
}


int sbplx_minimize(mot_float_type* model_parameters, /* in: initial guess, out: minimizer */
			       void* data,
			       const mot_float_type* const xstep0/* initial step sizes */){

    mot_float_type scratch[%(NMR_PARAMS)r * 2 // (xstep, delta_x)
                            + MAX_SUBSPACE_LENGTH * 2 // (subspace_model_parameters, subspace_xstep)
                            + (MAX_SUBSPACE_LENGTH+1)*(MAX_SUBSPACE_LENGTH+1) + 2*MAX_SUBSPACE_LENGTH // NMSimplex scratch
                       ];

    mot_float_type* xstep = scratch;
    mot_float_type* delta_x = xstep + %(NMR_PARAMS)r;
    mot_float_type* subspace_model_parameters = delta_x + %(NMR_PARAMS)r;
    mot_float_type* subspace_xstep = subspace_model_parameters + MAX_SUBSPACE_LENGTH;
    mot_float_type* nms_scratch = subspace_xstep + MAX_SUBSPACE_LENGTH;

    int x_indices[%(NMR_PARAMS)r]; /* permuted indices of model_parameters sorted by decreasing magnitude |delta_x| */
    int subspace_dimensions[%(NMR_PARAMS)r / MIN_SUBSPACE_LENGTH];

    int i, k;
    int itr;
    SubspaceData subspace_data;
    int nmr_subspaces;
    int subspace_starting_index;
    mot_float_type fdiff;
    mot_float_type fdiff_max;
    mot_float_type step_size_scale;
    mot_float_type stepnorm; // used in the computation of the next step size
    mot_float_type dxnorm;   // used in the computation of the next step size

    mot_float_type minf = evaluate(model_parameters, data);

    for(i = 0; i < %(NMR_PARAMS)r; i++){
        xstep[i] = xstep0[i];
        delta_x[i] = 0;
    }

    subspace_data.x_indices = x_indices;
    subspace_data.x = model_parameters;
    subspace_data.data = data;

    for(itr=0; itr < MAX_IT; itr++) {

        // first use delta_x to create the subspaces
        _sbplex_get_subspaces(delta_x, x_indices, subspace_dimensions, &nmr_subspaces, %(NMR_PARAMS)r,
                              MIN_SUBSPACE_LENGTH, MAX_SUBSPACE_LENGTH);


        // then use delta_x as a temporary container for the current parameters
        for(i = 0; i < %(NMR_PARAMS)r; i++){
            delta_x[i] = model_parameters[i];
        }

        // run NMSimplex on each subspace
        subspace_starting_index = 0; // loop variable, keeping track of the subspace index
        fdiff_max = 0; // records the largest gain in function value over the subspaces
        for(i = 0; i < nmr_subspaces; i++){

            // prepare the subspace data
            subspace_data.subspace_starting_index = subspace_starting_index;
            subspace_data.subspace_length = subspace_dimensions[i];

            // prepare the subspace parameters and step sizes (initial simplex scales)
            for(k = subspace_starting_index; k < subspace_dimensions[i] + subspace_starting_index; k++){
                subspace_model_parameters[k-subspace_starting_index] = model_parameters[x_indices[k]];
                subspace_xstep[k-subspace_starting_index] = xstep[x_indices[k]];
            }

            lib_nmsimplex(subspace_dimensions[i], subspace_model_parameters, (void*)&subspace_data, subspace_xstep,
                          &fdiff, PSI, %(PATIENCE_NMSIMPLEX)r * (subspace_dimensions[i] + 1),
                          ALPHA, BETA, GAMMA, DELTA, nms_scratch);

            // add the optimized subspace parameters to the current optimal set of model_parameters
            for(k = subspace_starting_index; k < subspace_dimensions[i] + subspace_starting_index; k++){
                model_parameters[x_indices[k]] = subspace_model_parameters[k-subspace_starting_index];
            }

            if(fdiff > fdiff_max){
                fdiff_max = fdiff;
            }

            // prepare for the next iteration
            subspace_starting_index += subspace_dimensions[i];
        }

        // compute change in optimal point, the previous delta_x contained the previous set of model parameters
        for (i = 0; i < %(NMR_PARAMS)r; ++i){
            delta_x[i] = model_parameters[i] - delta_x[i];
        }

        // stopping criteria using the infinity norm
        dxnorm = 0;
        stepnorm = 0;
        for (i = 0; i < %(NMR_PARAMS)r; ++i) {
            dxnorm = max(dxnorm, fabs(delta_x[i]));
            stepnorm = max(stepnorm, fabs(xstep[i]));
        }
        if(max(dxnorm, (mot_float_type)(stepnorm * PSI)) / max(stepnorm, (mot_float_type)1.0) <= USER_TOL_X){
            return 3;
        }

        // calculate the step size
        if(nmr_subspaces == 1){
            step_size_scale = PSI;
        }
        else {
            // calculate the L1 norm
            dxnorm = 0;
            stepnorm = 0;
            for (i = 0; i < %(NMR_PARAMS)r; ++i) {
                dxnorm += fabs(delta_x[i]);
                stepnorm += fabs(xstep[i]);
            }
            step_size_scale = min(max(dxnorm / stepnorm, (mot_float_type)OMEGA), (mot_float_type)(1.0/OMEGA));
        }

        // create the new step size (initial simplex scale)
        for (i = 0; i < %(NMR_PARAMS)r; ++i){
            if(delta_x[i] == 0){
                xstep[i] = -(xstep[i] * step_size_scale);
            }
            else{
                xstep[i] = copysign(xstep[i] * step_size_scale, delta_x[i]);
            }
        }
    }

    return 6;
}

int sbplex(mot_float_type* const model_parameters, void* data){
    const mot_float_type initial_simplex_scale[%(NMR_PARAMS)r] = %(INITIAL_SIMPLEX_SCALES)s;
    return sbplx_minimize(model_parameters, data, initial_simplex_scale);
}


#endif // SBPLEX_CL
