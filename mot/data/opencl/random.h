#ifndef RANDOM_H
#define RANDOM_H

/** The public interface for interacting with the random number generator.

These are meant to be used by the models and by the optimization / sampling routines.

Instantiating the rng_data needs to be done by the kernel using the RNG that is being used in MOT.

Afterwards, the rng_data can be used to create random numbers using any of the methods in this header.
*/

/** returns a uniform distributed double4 between 0 and 1 */
double4 rand4(void* rng_data);

/** returns a normal distributed double4 with mean 0 and sigma 1 */
double4 randn4(void* rng_data);

/** returns a uniform distributed float4 between 0 and 1 */
float4 frand4(void* rng_data);

/** returns a normal distributed float4 with mean 0 and sigma 1 */
float4 frandn4(void* rng_data);


/** returns a uniform distributed double between 0 and 1 */
double rand(void* rng_data);

/** returns a normal distributed double with mean 0 and sigma 1 */
double randn(void* rng_data);

/** returns a uniform distributed float between 0 and 1 */
float frand(void* rng_data);

/** returns a normal distributed float with mean 0 and sigma 1 */
float frandn(void* rng_data);

#endif // RANDOM_H
