/** The public interface for interacting with the random number generator.

Basic interaction with the RNG is as follows. First one creates an instance of the rng_data using the following code:

void* rng_data = create_rng_data(uint seed);

This rng_data can then be used to create random numbers using the methods:

- rand4 for a uniformdouble4
- randn4 for a

todo: make this an interface

*/



void* create_rng_data(uint seed){
    return (void*) &rand123_initialize_data_2key(counter);
}

void update_rng_data(void* rng_data){
    rand123_increment_loop_key((rand123_data*)rng_data);
}

double4 rand4(void* rng_data){
    return rand123_uniform_double4((rand123_data*)rng_data);
}

double4 randn4(void* rng_data){
    return rand123_normal_double4((rand123_data*)rng_data);
}

float4 frand4(void* rng_data){
    return rand123_uniform_float4((rand123_data*)rng_data);
}

float4 frandn4(void* rng_data){
    return rand123_normal_float4((rand123_data*)rng_data);
}

/** End of the random number generating functions */
