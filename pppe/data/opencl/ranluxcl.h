#ifndef RANLUXCL_H
#define RANLUXCL_H

#ifndef RANLUXCL_H_STATE_STRUCT
#define RANLUXCL_H_STATE_STRUCT
typedef struct{
	float4 s01to04;
	float4 s05to08;
	float4 s09to12;
	float4 s13to16;
	float4 s17to20;
	float4 s21to24;
	float4 carryin24stepnr; //Fourth component unused
} ranluxcl_state_t;
#endif

void ranluxcl_download_seed(ranluxcl_state_t *ranluxclstate, __global float4 *ranluxcltab);
void ranluxcl_upload_seed(ranluxcl_state_t *ranluxclstate, __global float4 *ranluxcltab);
float4 ranluxcl(ranluxcl_state_t *ranluxclstate);
void ranluxcl_initialization(int ins, global float4 *ranluxcltab);
float ranluxcl_gaussian(ranluxcl_state_t *ranluxclstate);
float2 ranluxcl_gaussian2(ranluxcl_state_t *ranluxclstate);
float4 ranluxcl_gaussian4(ranluxcl_state_t *ranluxclstate);

#endif //RANLUXCL_H
