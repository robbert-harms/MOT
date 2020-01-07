#ifndef _RAND123_DOT_CL
#define _RAND123_DOT_CL

/**
 * This CL file is the MOT interface to the rand123 library. It contains various random number generating functions
 * for generating random numbers in uniform and gaussian distributions.
 *
 * The rand123 supports various modes and precisions, in this front-end we have chosen for a precision of 4 words
 * of 32 bits (In rand123 terms, we have the 4x32 bit generators).
 */

/**
 * Generates the random bits used by the random functions.
 *
 * This will automatically update the state of the RNG to prepare for the next call.
 */
uint4 rand123_generate_bits(){
    ulong gid = (ulong)(get_global_id(0) / get_local_size(0));

    union {
        %(GENERATOR_NAME)s4x32_ctr_t rng_el;
        uint4 vec_el;
    } ctr, u;

    union {
        %(GENERATOR_NAME)s4x32_key_t rng_el;
        uint4 vec_el;
    } key;

    ctr.vec_el = __rng_state[gid];
    key.vec_el = __rng_state[gid + 4];

    u.rng_el = %(GENERATOR_NAME)s4x32(ctr.rng_el, key.rng_el);

    if (++__rng_state[gid + 0] == 0){
        if (++__rng_state[gid + 1] == 0){
            if (++__rng_state[gid + 2] == 0){
                ++__rng_state[gid + 3];
            }
        }
    }

    return u.vec_el;
}

/**
 * Applies the Box-Muller transformation on four uniformly distributed random numbers.
 *
 * This transforms uniform random numbers into Normal distributed random numbers.
 */
double4 rand123_box_muller_double4(double4 x){
    double r0 = sqrt(-2 * log(x.x));
    double c0;
    double s0 = sincos(((double) 2 * M_PI) * x.y, &c0);

    double r1 = sqrt(-2 * log(x.z));
    double c1;
    double s1 = sincos(((double) 2 * M_PI) * x.w, &c1);

    return (double4) (r0*c0, r0*s0, r1*c1, r1*s1);
}

float4 rand123_box_muller_float4(float4 x){
    float r0 = sqrt(-2 * log(x.x));
    float c0;
    float s0 = sincos(((double) 2 * M_PI) * x.y, &c0);

    float r1 = sqrt(-2 * log(x.z));
    float c1;
    float s1 = sincos(((double) 2 * M_PI) * x.w, &c1);

    return (float4) (r0*c0, r0*s0, r1*c1, r1*s1);
}
/** end of Box Muller transforms */

/** Random number generating functions in the Rand123 space */
double4 rand123_uniform_double4(){
    uint4 generated_bits = rand123_generate_bits();
    return ((double) (1/pown(2.0, 32))) * convert_double4(generated_bits) +
            ((double) (1/pown(2.0, 64))) * convert_double4(generated_bits);
}

double4 rand123_normal_double4(){
    return rand123_box_muller_double4(rand123_uniform_double4());
}

float4 rand123_uniform_float4(){
    uint4 generated_bits = rand123_generate_bits();
    return (float)(1/pown(2.0, 32)) * convert_float4(generated_bits);
}

float4 rand123_normal_float4(){
    return rand123_box_muller_float4(rand123_uniform_float4());
}
/** End of the random number generating functions */


double4 rand4(){
    return rand123_uniform_double4();
}

double4 randn4(){
    return rand123_normal_double4();
}

float4 frand4(){
    return rand123_uniform_float4();
}

float4 frandn4(){
    return rand123_normal_float4();
}

double rand(){
    return rand4().x;
}

double randn(){
    return randn4().x;
}

float frand(){
    return frand4().x;
}

float frandn(){
    return frandn4().x;
}

#endif // _RAND123_DOT_CL
