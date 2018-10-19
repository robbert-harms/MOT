#ifndef POWELL_CL
#define POWELL_CL

/**
 * Creator = Robbert Harms
 * Date = 2014-02-05
 * License = LGPL v3
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/**
   Uses the Powell's Quadratically Convergent Method of minimizing an objective function in a multidimensional space.

   This function is implemented in OpenCL by Robbert Harms, using the original Powell 1964 paper [1] and
   the Numerical Recipes chapter on Powell.

   It features two reset methods, the one from Numerical Recipes and the one Identity reset from Brent.
   The other reset method by Brent [2] could still be added, i.e. resetting to an orthogonal matrix found by singular
   value decomposition.

   References:

   [1] Powell, M. J. D. (1964). "An efficient method for finding the minimum of a function of several variables
        without calculating derivatives". Computer Journal. 7 (2): 155-162. doi:10.1093/comjnl/7.2.155.
   [2] Brent, Richard P. (1973). "Section 7.3: Powell's algorithm". Algorithms for minimization without derivatives.
        Englewood Cliffs, N.J.: Prentice-Hall. ISBN 0-486-41998-3.
*/

/* Used to set the maximum number of iterations to patience*(number_of_parameters+1). */
#define POWELL_EPSILON 30*MOT_EPSILON
#define POWELL_MAX_ITERATIONS (%(PATIENCE)r * (%(NMR_PARAMS)r+1))
#define POWELL_FUNCTION_TOLERANCE 30*MOT_EPSILON

#define MNBRACK_GOLD 1.618034 /* the default ratio by which successive intervals are magnified in Bracketing */
#define MNBRACK_GLIMIT 100.0 /* the maximum magnification allowed for a parabolic-fit step in Bracketing */
#define MNBRACK_EPSILON 30*MOT_EPSILON

#define BRENT_MAX_ITERATIONS (%(PATIENCE_LINE_SEARCH)r * (%(NMR_PARAMS)r+1))
#define BRENT_TOL 2 * 30 * MOT_EPSILON
#define BRENT_GOLD 0.3819660 /* golden ratio = (3 - sqrt(5))/2 */
#define BRENT_ZEPS 30 * MOT_EPSILON

/**
  * Set one of the reset methods. These are used to reset the search directions after a set number of steps to
  * prevent linear dependence between the search vectors
  */
#define POWELL_RESET_METHOD_RESET_TO_IDENTITY 0 /* Resets the search vectors to the I(nxn) matrix after every cycle of N (N = number of parameters) */
#define POWELL_RESET_METHOD_EXTRAPOLATED_POINT 1 /* see Numerical Recipes */
#define POWELL_RESET_METHOD POWELL_RESET_METHOD_%(RESET_METHOD)s


/**
 * A structure used to hold the data we are passing to the linear optimizer.
 *
 * The linear optimizer in turn should pass it to the linear evaluation function.
 */
typedef struct{
    local const mot_float_type* const point_0;
    local const mot_float_type* const point_1;
    local mot_float_type* tmp_point;
    void* data;
} linear_function_data;


/** The evaluation function we are expecting. */
double %(FUNCTION_NAME)s(local mot_float_type* x, void* data_void);


void mnbrack(mot_float_type* ax, mot_float_type* bx, mot_float_type* cx,
             mot_float_type* fa, mot_float_type* fb, mot_float_type* fc,
             void* eval_data);
int brent(mot_float_type ax, mot_float_type bx, mot_float_type cx,
          mot_float_type* xmin, mot_float_type* fmin, void* eval_data);


/**
 * Initializes the starting vectors.
 *
 * This fills the starting vector matrix with the identity matrix ensuring every vector is linearly independent.
 *
 * Args:
 *  search_directions (2d nxn array): the array with vectors to initialize)
 */
void powell_init_search_directions(local mot_float_type* search_directions){
    int i, j;
    for(i=0; i < %(NMR_PARAMS)r; i++){
        for(j=0; j < %(NMR_PARAMS)r; j++){
            search_directions[i * %(NMR_PARAMS)r + j] = (i == j ? 1.0 : 0.0);
        }
    }
}

/**
 * Checks if Powell should terminate
 *
 * Checks the stopping criteria. If the difference between the old function value and the new function value
 * is lower then a certain threshold, stop.
 *
 * Args:
 *  previous_fval: the previous function value
 *  new_fval: the new function value
 *
 * Returns:
 *  True if the optimizer should top, False otherwise
 */
bool powell_fval_diff_within_threshold(mot_float_type previous_fval, mot_float_type new_fval){
    return 2.0 * (previous_fval - new_fval) <= POWELL_FUNCTION_TOLERANCE * (fabs(previous_fval) + fabs(new_fval)) + POWELL_EPSILON;
}


/**
 * Finds the linear minimum on the line joining the first and the second data points.
 *
 * Suppose you have two points, ``p_0`` and ``p_1``, both in R^n.
 * This function tries to find the minimum function value on points on that line. The first point is supposed to be
 * fixed and we add to that a multiple of the second point. In other words, this uses a linear line search
 * to find the ``x`` that minimizes the function ``g(x) = f(p_0 + x * p_1)`` where f(y) is the function the user
 * tries to optimize with Powell and ``g(x)`` is the linear function we try to optimize in this function.
 *
 * Since OpenCL 1.2 does not have lambda expressions we can not use a lambda function to pass to the linear optimization
 * routine. To solve that we implement the linear optimizer here and set it up to use the
 * given ``powell_linear_data`` struct to pass the necessary data for optimization.
 *
 * Args:
 *  point_0: the static point
 *  point_1: the point we are moving towards
 *  tmp_point: a temporary storage array
 *
 * Modifies:
 *  point_0: set to ``p_0 + x * p_1``
 *  point_1: set to ``x * p_1``
 *
 * Returns:
 *  the function value at the optimum point found on the line.
 */
mot_float_type powell_find_linear_minimum(
        local mot_float_type* const point_0,
        local mot_float_type* const point_1,
        void* data,
        local mot_float_type* tmp_point){

    linear_function_data eval_data = {point_0, point_1, tmp_point, data};

    mot_float_type xmin;
    mot_float_type fval;
    mot_float_type ax, bx, cx, fa, fb, fc;

    mnbrack(&ax, &bx, &cx, &fa, &fb, &fc, (void*)&eval_data);
    brent(ax, bx, cx, &xmin, &fval, (void*)&eval_data);

    if(get_local_id(0) == 0){
        for(int j=0; j < %(NMR_PARAMS)r; j++){
            point_1[j] *= xmin;
            point_0[j] += point_1[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return fval;
}


/**
 * The linear evaluation function used by the 1d line optimization routine.
 *
 * For its usage and reason of existence please check the docs of the function :ref:`powell_find_linear_minimum`.
 *
 * Args:
 *  x: the point at which to evaluate the function
 *  eval_data: the data used to evaluate the function.
 *
 * Returns:
 *  the function value at the given point
 */
double powell_linear_eval_function(mot_float_type x, void* eval_data){

    linear_function_data f_data = *((linear_function_data*)eval_data);
    local mot_float_type* xt = ((linear_function_data*)eval_data)->tmp_point;

    if(get_local_id(0) == 0){
        for(int j=0; j < %(NMR_PARAMS)r; j++){
            xt[j] = f_data.point_0[j] + x * f_data.point_1[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return %(FUNCTION_NAME)s(xt, f_data.data);
}


/**
 * Do the line searches. This is the first step in the basic procedure in Powell (1964).
 *
 * This loops through the search vectors and finds for every search vector the point with the lowest function
 * value between the starting point and the search vector. During this process the starting point for the next
 * iteration is set to the optimum value of the current iteration.
 *
 * Args:
 *   starting_point: the starting point for the search
 *   search_directions: the nxn array with search directions to loop through
 *   data: the evaluation data
 *   fval: the current best known function value
 *   largest_decrease: -
 *   index_largest_decrease: -
 *
 * Modifies:
 *   starting_point: set to the position of the new lowest function point
 *   largest_decrease: set to the search vector that yielded the largest decrease with respect to
 *      the at the time best found function value.
 *   index_largest_decrease: the index of the search vector that yielded the largest decrease
 *
 * Returns:
 *   the new lowest function value
 */
mot_float_type powell_do_line_searches(
        local mot_float_type* search_directions,
        void* data,
        mot_float_type fval,
        local mot_float_type* starting_point,
        mot_float_type* largest_decrease,
        int* index_largest_decrease,
        local mot_float_type* tmp_point,
        local mot_float_type* tmp_point2){

    int i, j;

    *largest_decrease = 0.0;
    *index_largest_decrease = 0;

    mot_float_type fval_previous;

    for(i = 0; i < %(NMR_PARAMS)r; i++){
        if(get_local_id(0) == 0){
            for(j = 0; j < %(NMR_PARAMS)r; j++){
                tmp_point[j] = search_directions[j * %(NMR_PARAMS)r + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        fval_previous = fval;
        fval = powell_find_linear_minimum(starting_point, tmp_point, data, tmp_point2);

        if(fabs(fval_previous - fval) > *largest_decrease){
            *largest_decrease = fabs(fval_previous - fval);
            *index_largest_decrease = i;
        }
    }
    return fval;
}

#if POWELL_RESET_METHOD == POWELL_RESET_METHOD_EXTRAPOLATED_POINT
/**
 * Evaluate the problem function at an extrapolated point lying between the best point found and the old point.
 *
 * This is a method described in Numerical Recipes as part of a way to prevent linear dependency of the search vectors.
 * We extrapolate the best point by a factor of two and subtract from that the old point. The new point is
 * evaluated and the resulting function value is returned.
 *
 * Args:
 *  new_best_point: the currently found best point
 *  old_point: the old point
 *  data: problem data
 *
 * Returns:
 *  the function value at the extrapolated point
 */
mot_float_type powell_evaluate_extrapolated(local mot_float_type* new_best_point,
                                            local mot_float_type* old_point,
                                            void* data,
                                            local mot_float_type* tmp_point){
    if(get_local_id(0) == 0){
        for(int i = 0; i < %(NMR_PARAMS)r; i++){
            tmp_point[i] = 2.0 * new_best_point[i] - old_point[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return %(FUNCTION_NAME)s(tmp_point, data);
}

/**
 * Test if Powell should exchange the search directions or not.
 *
 * This is the test in Numerical Recipes, other tests can also be used for the other methods.
 */
bool powell_should_exchange_search_directions(
        mot_float_type fval_at_start_of_iteration,
        mot_float_type fval_best_found,
        mot_float_type fval_extrapolated,
        mot_float_type largest_decrease){

    if(fval_extrapolated >= fval_at_start_of_iteration){
        return false;
    }

    if((2.0 * (fval_at_start_of_iteration - 2.0 * fval_best_found + fval_extrapolated)
            * (pown(fval_at_start_of_iteration - fval_best_found - largest_decrease, 2)))
            >= largest_decrease * pown(fval_at_start_of_iteration - fval_extrapolated, 2)){
        return false;
    }
    return true;
}

#define SHOULD_EXCHANGE_SEARCH_DIRECTION powell_should_exchange_search_directions(fval_at_start_of_iteration, fval, fval_extrapolated, largest_decrease)

#elif POWELL_RESET_METHOD == POWELL_RESET_METHOD_RESET_TO_IDENTITY
#define SHOULD_EXCHANGE_SEARCH_DIRECTION true
#endif

int powell(local mot_float_type* model_parameters, void* data,
           local mot_float_type* scratch_mot_float_type){

    local mot_float_type* scratch_ind = scratch_mot_float_type;
    local mot_float_type* parameters_at_start_of_iteration = scratch_ind;  scratch_ind += %(NMR_PARAMS)r;
    local mot_float_type* search_directions = scratch_ind;                 scratch_ind += %(NMR_PARAMS)r * %(NMR_PARAMS)r;
    local mot_float_type* tmp_point = scratch_ind;                         scratch_ind += %(NMR_PARAMS)r;
    local mot_float_type* tmp_point2 = scratch_ind;                        scratch_ind += %(NMR_PARAMS)r;

    mot_float_type fval, fval_extrapolated, fval_at_start_of_iteration, largest_decrease;

    int i, j, index_largest_decrease;
    int iteration = 0;

    if(get_local_id(0) == 0){
        powell_init_search_directions(search_directions);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    fval = %(FUNCTION_NAME)s(model_parameters, data);

    while(iteration++ < POWELL_MAX_ITERATIONS){
        fval_at_start_of_iteration = fval;

        if(get_local_id(0) == 0){
            for(i=0; i < %(NMR_PARAMS)r; i++){
                parameters_at_start_of_iteration[i] = model_parameters[i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        fval = powell_do_line_searches(search_directions, data, fval, model_parameters,
                                       &largest_decrease, &index_largest_decrease, tmp_point, tmp_point2);

        if(powell_fval_diff_within_threshold(fval_at_start_of_iteration, fval)){
            return 1;
        }

        #if POWELL_RESET_METHOD == POWELL_RESET_METHOD_EXTRAPOLATED_POINT
            fval_extrapolated = powell_evaluate_extrapolated(model_parameters, parameters_at_start_of_iteration, data, tmp_point);
        #endif

        if(SHOULD_EXCHANGE_SEARCH_DIRECTION){

            if(get_local_id(0) == 0){
                #if POWELL_RESET_METHOD == POWELL_RESET_METHOD_EXTRAPOLATED_POINT
                    for(i = 0; i < %(NMR_PARAMS)r; i++){
                        // remove the one with the largest increase (see Numerical Recipes)
                        search_directions[i * %(NMR_PARAMS)r + index_largest_decrease] =
                            search_directions[i * %(NMR_PARAMS)r + %(NMR_PARAMS)r-1];

                        // add p_n - p_0, see Powell 1964.
                        search_directions[i * %(NMR_PARAMS)r + %(NMR_PARAMS)r-1] =
                            model_parameters[i] - parameters_at_start_of_iteration[i];
                    }

                #elif POWELL_RESET_METHOD == POWELL_RESET_METHOD_RESET_TO_IDENTITY
                    if((iteration + 1) %% %(NMR_PARAMS)r == 0){
                        powell_init_search_directions(search_directions);
                    }
                    else{
                        for(i = 0; i < %(NMR_PARAMS)r; i++){
                            for(j = 0; j < %(NMR_PARAMS)r - 1; j++){
                                search_directions[i * %(NMR_PARAMS)r + j] = search_directions[i * %(NMR_PARAMS)r + j+1];
                            }
                            // add p_n - p_0, see Powell 1964.
                            search_directions[i * %(NMR_PARAMS)r + %(NMR_PARAMS)r-1] =
                                model_parameters[i] - parameters_at_start_of_iteration[i];
                        }
                    }
                #endif

                // this uses ``parameters_at_start_of_iteration`` to find the last function minimum, this saves an array
                for(i = 0; i < %(NMR_PARAMS)r; i++){
                    parameters_at_start_of_iteration[i] = model_parameters[i] - parameters_at_start_of_iteration[i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            fval = powell_find_linear_minimum(model_parameters, parameters_at_start_of_iteration, data, tmp_point);
        }
    }
    return 6;
}

/**
 * Bracket the minimum of a function.
 */
void mnbrack(mot_float_type* ax, mot_float_type* bx, mot_float_type* cx,
             mot_float_type* fa, mot_float_type* fb, mot_float_type* fc,
             void* eval_data){
    *ax = 0.0;
    *bx = 1.0;
    *fa = 0.0;
    *fb = 0.0;
    *fc = 0.0;

    mot_float_type ulim, u, r, q, fu, tmp;
    mot_float_type maxarg = 0.0;

    *fa = powell_linear_eval_function(*ax, eval_data);
    *fb = powell_linear_eval_function(*bx, eval_data);

    if(*fb > *fa){
        tmp = *bx;
        *bx = *ax;
        *ax = tmp;

        tmp = *fb;
        *fb = *fa;
        *fa = tmp;
    }

    *cx = *bx + MNBRACK_GOLD * (*bx - *ax);
    *fc = powell_linear_eval_function(*cx, eval_data);

    while(*fb > *fc){
        r = (*bx - *ax) * (*fb - *fc);
        q = (*bx - *cx) * (*fb - *fa);

        maxarg = fmax(fabs(q-r), (mot_float_type)MNBRACK_EPSILON);

        u = (*bx) - ((*bx - *cx) * q - (*bx - *ax) * r) / (2.0 * copysign(maxarg, q-r));
        ulim = (*bx) + MNBRACK_GLIMIT * (*cx - *bx);

        if((*bx - u) * (u - *cx) > 0.0){
            fu = powell_linear_eval_function(u, eval_data);

            if(fu < *fc){
                *ax = *bx;
                *bx = u;
                *fa = *fb;
                *fb = fu;
                break;
            }
            else if(fu > *fb){
                *cx = u;
                *fc = fu;
                break;
            }
            u = (*cx) + MNBRACK_GOLD * (*cx - *bx);
            fu = powell_linear_eval_function(u, eval_data);
        }
        else if((*cx - u) * (u - ulim) > 0.0){
            fu = powell_linear_eval_function(u, eval_data);

            if(fu < *fc){
                *bx = *cx;
                *cx = u;
                u = *cx + MNBRACK_GOLD * (*cx - *bx);

                *fb = *fc;
                *fc = fu;
                fu = powell_linear_eval_function(u, eval_data);
            }
        }
        else if((u - ulim) * (ulim - *cx) >= 0.0){
            u = ulim;
            fu = powell_linear_eval_function(u, eval_data);
        }
        else{
            u = (*cx) + MNBRACK_GOLD * (*cx - *bx);
            fu = powell_linear_eval_function(u, eval_data);
        }
        *ax = *bx;
        *bx = *cx;
        *cx = u;

        *fa = *fb;
        *fb = *fc;
        *fc = fu;
    }
}

/**
 * Line search using Brent's method.
 * Given a function f, and given a bracketing triplet of abscissas ax, bx, cx (such that bx is between ax and cx,
 * and f(bx) is less than both f(ax) and (cx)), this routine isolates the minimum to a fractional precision of about
 * tol using Brent's method. The abscissa of the minimum is returned as xmin, and the minimum function value is
 * returned as fmin.
 *
 * The return value signifies the return code.
 */
int brent(mot_float_type ax, mot_float_type bx, mot_float_type cx, mot_float_type* xmin,
          mot_float_type* fmin, void* eval_data){
    mot_float_type u, r, q, fu, tmp;
    mot_float_type d, fx, fv, fw;
    mot_float_type p, tol1, tol2, v, w, x, xm;
    mot_float_type e=0.0;

    mot_float_type a=(ax < cx ? ax : cx);
    mot_float_type b=(ax > cx ? ax : cx);

    x=w=v=bx;
    fw=fv=fx=powell_linear_eval_function(x, eval_data);

    for(uint iter = 0; iter < BRENT_MAX_ITERATIONS; iter++){
        xm = 0.5 * (a + b);
        tol1 = BRENT_TOL * fabs(x) + BRENT_ZEPS;
        tol2 = 2.0 * tol1;

        if(fabs(x - xm) <= (tol2 - 0.5 * (b - a))){
            *xmin = x;
            *fmin = fx;
            return 0;
        }

        if(fabs(e) > tol1){
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);

            if(q > 0.0){
                p = -p;
            }

            q = fabs(q);
            tmp = e;
            e = d;

            // Explicit checking that the three points used to fit the quadratic are not in fact co-linear
            // http://numerical.recipes/forum/showthread.php?p=4893 (2018-05-14).
            if(fabs(p) < BRENT_TOL || fabs(p) >= fabs(0.5 * q * tmp) || p <= q*(a-x) || p >= q*(b-x)){
                e = (x >= xm ? a : b) - x;
                d = BRENT_GOLD * e;
            }
            else {
                d = p / q;
                u = x + d;
                if(u - a < tol2 || b - u < tol2){
                    d = copysign(tol1, xm - x);
                }
            }
        }
        else{
            e = (x >= xm ? a : b) - x;
            d = BRENT_GOLD * e;
        }

        u = (fabs(d) >= tol1 ? x + d : x + copysign(tol1, d));
        fu = powell_linear_eval_function(u, eval_data);

        if(fu <= fx){
            if(u >= x){
                a=x;
            }
            else{
                b=x;
            }
            v = w;
            w = x;
            x = u;

            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if(u < x){
                a=u;
            }
            else{
                b=u;
            }
            if (fu <= fw || w == x) {
                v=w;
                w=u;
                fv=fw;
                fw=fu;
            }
            else if(fu <= fv || v == x || v == w){
                v=u;
                fv=fu;
            }
        }
    }
    *xmin=x;
    *fmin = fx;
    return 0;
}

#undef BRENT_MAX_ITERATIONS
#undef BRENT_TOL
#undef BRENT_GOLD
#undef BRENT_ZEPS

#undef MAX_ITERATIONS
#undef MNBRACK_GOLD
#undef MNBRACK_GLIMIT
#undef MNBRACK_EPSILON
#undef POWELL_FUNCTION_TOLERANCE
#undef POWELL_EPSILON
#undef POWELL_RESET_METHOD_RESET_TO_IDENTITY
#undef POWELL_RESET_METHOD_EXTRAPOLATED_POINT
#undef POWELL_RESET_METHOD
#undef SHOULD_EXCHANGE_SEARCH_DIRECTION

#endif // POWELL_CL
