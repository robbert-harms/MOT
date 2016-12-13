#ifndef _RAND123_DOT_CL
#define _RAND123_DOT_CL

/**
 * The following are the functions that one should actually use in MOT applications.
 *
 * It contains various random number generating functions for generating random numbers in:
 * - uniform [0,1] distribution
 * - gaussian with mean 0 and std 1.
 */

/**
 * The information needed by the random functions to generate unique random numbers.
 */
typedef struct{
    %(GENERATOR_FUNCTION)s_ctr_t counter;
    %(GENERATOR_FUNCTION)s_key_t key;
} rand123_data;

/**
 * Generates the random bits used by the random functions
 */
uint4 rand123_generate_bits(rand123_data* rng_data){

    %(GENERATOR_FUNCTION)s_ctr_t* ctr = &rng_data->counter;
    %(GENERATOR_FUNCTION)s_key_t* key = &rng_data->key;

    union {
        %(GENERATOR_FUNCTION)s_ctr_t ctr_el;
        uint4 vec_el;
    } u;

    u.ctr_el = %(GENERATOR_FUNCTION)s(*ctr, *key);
    return u.vec_el;
}

/**
 * Initializes the rand123_data structure without an external key.
 *
 * The first element in the key is automatically set to the global id of the kernel,
 * and the second element is set by default to 0. We therefore need to give no key data.
 */
rand123_data rand123_initialize_data(uint counter[4]){
    %(GENERATOR_FUNCTION)s_ctr_t k = {{get_global_id(0), 0}};
    %(GENERATOR_FUNCTION)s_key_t c = {{(uint32_t)*counter}};

    rand123_data rng_data = {c, k};
    return rng_data;
}

/**
 * Initializes the rand123_data structure using constant memory without an external key.
 *
 * The same function as ``rand123_initialize_data`` but this accepts ``constant`` memory pointers.
 */
rand123_data rand123_initialize_data_constmem(constant uint counter[4]){
    return rand123_initialize_data((uint []){counter[0], counter[1], counter[2], counter[3]});
}


/**
 * Initializes the rand123_data structure with additional precision (extra key).
 *
 * The first element in the key is automatically set to the global id of the kernel,
 * and the second element is set by default to 0. We therefore need only to set the other two
 * key elements.
 */
rand123_data rand123_initialize_data_extra_precision(uint counter[4], uint key[2]){
    %(GENERATOR_FUNCTION)s_ctr_t k = {{get_global_id(0), 0, key[0], key[1]}};
    %(GENERATOR_FUNCTION)s_key_t c = {{(uint32_t)*counter}};

    rand123_data rng_data = {c, k};
    return rng_data;
}

/**
 * Initializes the rand123_data structure with additional precision (extra key) using constant memory pointers.
 *
 * The same function as ``rand123_initialize_data_extra_precision`` but this accepts ``constant`` memory pointers.
 */
rand123_data rand123_initialize_data_extra_precision_constmem(constant uint counter[4], constant uint key[2]){
    return rand123_initialize_data_extra_precision(
        (uint []){counter[0], counter[1], counter[2], counter[3]},
        (uint []){key[0], key[1]});
}

/**
 * Sets the second element of the key to the given counter.
 *
 * One needs to call this function after every call to a random number generating function
 * to ensure the next number will be different.
 */
void rand123_set_loop_key(rand123_data* rng_data, int key){
    rng_data->key.v[1] = key;
}

/**
 * Increments the value in the second key element by one.
 *
 * One needs to call this function after every call to a random number generating function
 * to ensure the next number will be different.
 */
void rand123_increment_loop_key(rand123_data* rng_data){
    rng_data->key.v[1]++;
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
double4 rand123_uniform_double4(rand123_data* rng_data){
    uint4 generated_bits = rand123_generate_bits(rng_data);
    return ((double) (1/pown(2.0, 32))) * convert_double4(generated_bits) +
        ((double) (1/pown(2.0, 64))) * convert_double4(generated_bits);
}

double4 rand123_normal_double4(rand123_data* rng_data){
    return rand123_box_muller_double4(rand123_uniform_double4(rng_data));
}

float4 rand123_uniform_float4(rand123_data* rng_data){
    uint4 generated_bits = rand123_generate_bits(rng_data);
    return (float)(1/pown(2.0, 32)) * convert_float4(generated_bits);
}

float4 rand123_normal_float4(rand123_data* rng_data){
    return rand123_box_muller_float4(rand123_uniform_float4(rng_data));
}
/** End of the random number generating functions */


/** Implementations of the random.h header */
double4 rand4(void* rng_data){
    double4 val = rand123_uniform_double4((rand123_data*)rng_data);
    rand123_increment_loop_key((rand123_data*)rng_data);
    return val;
}

double4 randn4(void* rng_data){
    double4 val = rand123_normal_double4((rand123_data*)rng_data);
    rand123_increment_loop_key((rand123_data*)rng_data);
    return val;
}

float4 frand4(void* rng_data){
    float4 val = rand123_uniform_float4((rand123_data*)rng_data);
    rand123_increment_loop_key((rand123_data*)rng_data);
    return val;
}

float4 frandn4(void* rng_data){
    float4 val = rand123_normal_float4((rand123_data*)rng_data);
    rand123_increment_loop_key((rand123_data*)rng_data);
    return val;
}

double rand(void* rng_data){
    return rand4(rng_data).x;
}

double randn(void* rng_data){
    return randn4(rng_data).x;
}

float frand(void* rng_data){
    return frand4(rng_data).x;
}

float frandn(void* rng_data){
    return frandn4(rng_data).x;
}

#endif // _RAND123_DOT_CL
