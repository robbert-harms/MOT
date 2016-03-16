#ifndef RANLUXCL_H
#define RANLUXCL_H

typedef struct{
	float
		s01, s02, s03, s04,
		s05, s06, s07, s08,
		s09, s10, s11, s12,
		s13, s14, s15, s16,
		s17, s18, s19, s20,
		s21, s22, s23, s24;
	float carry;
	float dummy; //Causes struct to be a multiple of 128 bits
	int in24;
	int stepnr;
} ranluxcl_state_t;

//Initial prototypes makes Apple's compiler happy
void ranluxcl_download_seed(ranluxcl_state_t *, global ranluxcl_state_t *);
void ranluxcl_upload_seed(ranluxcl_state_t *, global ranluxcl_state_t *);
float ranluxcl_os(float, float, float *, float *);
float4 ranluxcl32(ranluxcl_state_t *);
void ranluxcl_synchronize(ranluxcl_state_t *);
void ranluxcl_initialization(uint, global ranluxcl_state_t *);
float4 ranluxcl32norm(ranluxcl_state_t *);

float ranluxcl_gaussian(ranluxcl_state_t *ranluxclstate);
float2 ranluxcl_gaussian2(ranluxcl_state_t *ranluxclstate);
float4 ranluxcl_gaussian4(ranluxcl_state_t *ranluxclstate);
float rand(void * rand_settings);

#endif //RANLUXCL_H
