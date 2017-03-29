#ifndef NMSIMPLEX_CL
#define NMSIMPLEX_CL

/**
 * Author = Robbert Harms
 * Date = 2014-09-29
 * License = see hereunder
 * Maintainer = Robbert Harms
 * Email = robbert.harms@maastrichtuniversity.nl
 */

/*
 * Program: nmsimplex.c
 * Author : Michael F. Hutt
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
 * Removed constraints since MOT features parameter transformations
 */

#define PATIENCE %(PATIENCE)r                  /* Used to set the maximum number of iterations
                                                   to patience * (number_of_parameters + 1). */
#define MAX_IT (PATIENCE * (%(NMR_PARAMS)r+1))
#define ALPHA       %(ALPHA)r                   /* reflection coefficient, default 1 */
#define BETA        %(BETA)r                    /* contraction coefficient, default 0.5 */
#define GAMMA       %(GAMMA)r                   /* expansion coefficient default 2 */
#define DELTA       %(DELTA)r                   /* reduction coefficient default 0.5 */
#define USER_TOL_X     30*MOT_EPSILON              /** the precision we break at*/

int nmsimplex(mot_float_type* const model_parameters, const void* const data){

    int return_code = 6; /** the default return code is that we exhausted our patience */
	int vs;         /* vertex with smallest value */
	int vh;         /* vertex with next smallest value */
	int vg;         /* vertex with largest value */
    int i, j;        /** helper variables */
	int itr;	      /* track the number of iterations */
	double tmp;

	mot_float_type fr;      /* value of function at reflection point */
	mot_float_type fe;   /* value of function at expansion point */
	mot_float_type fc;      /* value of function at contraction point */

	mot_float_type vm[%(NMR_PARAMS)r]; /* centroid - coordinates */
	mot_float_type vr[%(NMR_PARAMS)r]; /* reflection - coordinates */
    mot_float_type ve_vc[%(NMR_PARAMS)r]; /* expansion - coordinates, & contraction - coordinates,
     										 that is, we use this variable at two points for different purposes.*/

    mot_float_type vertices[%(NMR_PARAMS)r + 1][%(NMR_PARAMS)r];     /* holds vertices of simplex */
    mot_float_type func_vals[%(NMR_PARAMS)r + 1]; /* value of function at each vertex */

    /** the scale of the initial simplex, should be set by python code as a string: {v1, v2, ...} */
    mot_float_type simplex_scale[%(NMR_PARAMS)r] = %(INITIAL_SIMPLEX_SCALES)s;

    /*
     * Create the initial simplex.
	 * We assume one of the vertices is 0,0
	 * Furthermore we set x_0 = x_input to allow for proper restarts.
	 */
	for (i=0;i<%(NMR_PARAMS)r;i++) {
		vertices[0][i] = model_parameters[i];
	}
	for (i=1;i<=%(NMR_PARAMS)r;i++) {
		for (j=0;j<%(NMR_PARAMS)r;j++) {
			vertices[i][j] = sqrt(%(NMR_PARAMS)r + 1.0) - 1;
			if (i-1 == j){
			    vertices[i][j] += %(NMR_PARAMS)r;
			}
            vertices[i][j] /= (%(NMR_PARAMS)r * M_SQRT2);
            /** vertices now contains the unit vector e_j in R^n. */

            /** set x_j = x_input + h_j * e_j */
            vertices[i][j] = model_parameters[j] + simplex_scale[i-1] * vertices[i][j];
		}
	}

	/* find the initial function values */
	for (j=0;j<=%(NMR_PARAMS)r;j++) {
		func_vals[j] = evaluate(vertices[j], data);
	}

	/* begin the main loop of the minimization */
	for (itr=0; itr <= MAX_IT; itr++) {
		/* find the index of the largest and smallest value */
		vg=0;
		vs=0;
		for (j=0;j<=%(NMR_PARAMS)r;j++) {
            /* find largest */
			if (func_vals[j] > func_vals[vg]) {
				vg = j;
			}

		    /* find smallest */
			if (func_vals[j] < func_vals[vs]) {
				vs = j;
			}
		}

		/* find the index of the second largest value */
		vh=vs;
		for (j=0;j<=%(NMR_PARAMS)r;j++) {
			if (func_vals[j] > func_vals[vh] && func_vals[j] < func_vals[vg]) {
				vh = j;
			}
		}

		/* calculate the centroid */
		for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
			tmp=0.0;
			for (i=0;i<=%(NMR_PARAMS)r;i++) {
				if (i!=vg) {
					tmp += vertices[i][j];
				}
			}
			vm[j] = tmp/%(NMR_PARAMS)r;
		}

		/* reflect vg to new vertex vr */
		for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
			vr[j] = vm[j] + ALPHA * (vm[j] - vertices[vg][j]);
		}
		fr = evaluate(vr, data);

		if (fr < func_vals[vh] && fr >= func_vals[vs]) {
			for (j=0; j <= %(NMR_PARAMS)r-1; j++){
				vertices[vg][j] = vr[j];
			}
			func_vals[vg] = fr;
		}

		/* investigate a step further in this direction */
		if(fr < func_vals[vs]){
			for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
				/** ve_vc here used as ve */
				ve_vc[j] = vm[j] + GAMMA * (vr[j] - vm[j]);
			}

			fe = evaluate(ve_vc, data);
			if (fe < fr){
				for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
					vertices[vg][j] = ve_vc[j];
				}
				func_vals[vg] = fe;
			}
			else {
				for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
					vertices[vg][j] = vr[j];
				}
				func_vals[vg] = fr;
			}
		}

		/* check to see if a contraction is necessary */
		if (fr >= func_vals[vh]) {
			if (fr < func_vals[vg] && fr >= func_vals[vh]) {
				/* perform outside contraction */
				for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
					/** ve_vc here used as vc */
					ve_vc[j] = vm[j] + BETA * (vr[j]-vm[j]);
				}
			}
			else {
				/* perform inside contraction */
				for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
					ve_vc[j] = vm[j] - BETA * (vm[j] - vertices[vg][j]);
				}
			}

			fc = evaluate(ve_vc, data);
			if (fc < func_vals[vg]) {
				for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
					vertices[vg][j] = ve_vc[j];
				}
				func_vals[vg] = fc;
			}
			else {
                /* at this point the contraction is not successful,
                   we must reduce (by default halve) the distance from vs to all the
                   vertices of the simplex and then continue.
                */
				for (i=0;i<=%(NMR_PARAMS)r;i++) {
					if (i != vs) {
						for (j=0;j<=%(NMR_PARAMS)r-1;j++) {
							vertices[i][j] = vertices[vs][j] + (vertices[i][j]-vertices[vs][j]) * DELTA;
						}
					}
				}

				func_vals[vg] = evaluate(vertices[vg], data);
				func_vals[vh] = evaluate(vertices[vh], data);
			}
		}

		/* test for convergence */
		tmp = 0.0;
		for (j=0;j<=%(NMR_PARAMS)r;j++) {
			tmp += func_vals[j];
		}
		/** fr here used as tmp dummy */
		fr = tmp/(%(NMR_PARAMS)r+1);

		tmp = 0.0;
		for (j=0;j<=%(NMR_PARAMS)r;j++) {
			tmp += ((func_vals[j]-fr) * (func_vals[j]-fr)) / (%(NMR_PARAMS)r);
		}
		tmp = sqrt(tmp);

		if (tmp < USER_TOL_X){
		    return_code = 1;
		    break;
		}
	}
	/* end main loop of the minimization */

	/* find the index of the largest and smallest value */
	vs = 0;
	for (j=0;j<=%(NMR_PARAMS)r;j++) {
		/* find smallest */
        if (func_vals[j] < func_vals[vs]) {
            vs = j;
        }
	}

    for (j=0;j<%(NMR_PARAMS)r;j++) {
		model_parameters[j] = vertices[vs][j];
	}

	return return_code;
}

#undef PATIENCE
#undef MAX_IT
#undef ALPHA
#undef BETA
#undef GAMMA
#undef USER_TOL_X

#endif // NMSIMPLEX_CL
