//
// Program to perform ADI time-marching on a regular 3D grid
// Alternating Direction Implicit Method

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "cutil_inline.h"
#include <stdio.h>

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
// define kernel block size for 
////////////////////////////////////////////////////////////////////////

#define BLOCK_X 32
#define BLOCK_Y 8
// device code


__global__ void GPU_adi_rhs(int NX, int NY, int NZ, float lam,
	const float* __restrict__ d_u,
	float* __restrict__ d_du,
	float* __restrict__ d_ax,
	float* __restrict__ d_bx,
	float* __restrict__ d_cx,
	float* __restrict__ d_ay,
	float* __restrict__ d_by,
	float* __restrict__ d_cy,
	float* __restrict__ d_az,
	float* __restrict__ d_bz,
	float* __restrict__ d_cz)
{
	int   i, j, k, indg, active;
	float du, a, b, c;

	int NXM1 = NX - 1;
	int NYM1 = NY - 1;
	int NZM1 = NZ - 1;

#define IOFF 1
#define JOFF NX
#define KOFF NX*NY

	//
	// set up indices for main block
	//

	i = threadIdx.x + blockIdx.x*BLOCK_X;
	j = threadIdx.y + blockIdx.y*BLOCK_Y;
	indg = i + j*NX;

	active = (i<NX) && (j<NY);

	//
	// loop over k-planes
	//

	for (k = 0; k<NZ; k++) {

		//
		// calculate r.h.s. and set a,b,c, coefficients
		//

		if (active) {
			if (i == 0 || i == NXM1 || j == 0 || j == NYM1 || k == 0 || k == NZM1) {
				du = 0.0f;          // Dirichlet b.c.'s
				a = 0.0f;
				b = 1.0f;
				c = 0.0f;
			}
			else {
				du = lam * (d_u[indg - IOFF] + d_u[indg + IOFF]
					+ d_u[indg - JOFF] + d_u[indg + JOFF]
					+ d_u[indg - KOFF] + d_u[indg + KOFF] - 6.0f*d_u[indg]);
				a = -0.5f*lam;
				b = 1.0f + lam;
				c = -0.5f*lam;
			}

			d_du[indg] = du;
			d_ax[indg] = a;
			d_bx[indg] = b;
			d_cx[indg] = c;
			d_ay[indg] = a;
			d_by[indg] = b;
			d_cy[indg] = c;
			d_az[indg] = a;
			d_bz[indg] = b;
			d_cz[indg] = c;

			indg += KOFF;
		}
	}
}

__global__ void GPU_adi_x_float4(int NX, int NY, int NZ,
	const float4* __restrict__ d_a,
	const float4* __restrict__ d_b,
	const float4* __restrict__ d_c,
	float4* __restrict__ d_d) {
	int    i, j, k, indg;
	float  aa, bb, cc, dd, c2[256], d2[256];
	float4 a4, b4, c4, d4;

	//
	// set up indices for main block
	//

	j = threadIdx.x + blockIdx.x*blockDim.x;
	k = threadIdx.y + blockIdx.y*blockDim.y;
	indg = NX*(j + k*NY) / 4;

	if ((j<NY) && (k<NZ)) {

		//
		// forward pass
		//

		a4 = d_a[indg];
		b4 = d_b[indg];
		c4 = d_c[indg];
		d4 = d_d[indg];

		bb = 1.0f / b4.x;
		cc = bb   * c4.x;
		dd = bb   * d4.x;
		c2[0] = cc;
		d2[0] = dd;

		aa = a4.y;
		bb = b4.y - aa*cc;
		dd = d4.y - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4.y;
		dd = bb*dd;
		c2[1] = cc;
		d2[1] = dd;

		aa = a4.z;
		bb = b4.z - aa*cc;
		dd = d4.z - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4.z;
		dd = bb*dd;
		c2[2] = cc;
		d2[2] = dd;

		aa = a4.w;
		bb = b4.w - aa*cc;
		dd = d4.w - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4.w;
		dd = bb*dd;
		c2[3] = cc;
		d2[3] = dd;

		for (i = 4; i<NX; i += 4) {
			indg = indg + 1;
			a4 = d_a[indg];
			b4 = d_b[indg];
			c4 = d_c[indg];
			d4 = d_d[indg];

			aa = a4.x;
			bb = b4.x - aa*cc;
			dd = d4.x - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.x;
			dd = bb*dd;
			c2[i] = cc;
			d2[i] = dd;

			aa = a4.y;
			bb = b4.y - aa*cc;
			dd = d4.y - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.y;
			dd = bb*dd;
			c2[i + 1] = cc;
			d2[i + 1] = dd;

			aa = a4.z;
			bb = b4.z - aa*cc;
			dd = d4.z - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.z;
			dd = bb*dd;
			c2[i + 2] = cc;
			d2[i + 2] = dd;

			aa = a4.w;
			bb = b4.w - aa*cc;
			dd = d4.w - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.w;
			dd = bb*dd;
			c2[i + 3] = cc;
			d2[i + 3] = dd;
		}

		//
		// reverse pass
		//

		d4.w = dd;
		dd = d2[NX - 2] - c2[NX - 2] * dd;
		d4.z = dd;
		dd = d2[NX - 3] - c2[NX - 3] * dd;
		d4.y = dd;
		dd = d2[NX - 4] - c2[NX - 4] * dd;
		d4.x = dd;

		d_d[indg] = d4;

		for (i = NX - 5; i >= 0; i -= 4) {
			indg = indg - 1;
			dd = d2[i] - c2[i] * dd;
			d4.w = dd;
			dd = d2[i - 1] - c2[i - 1] * dd;
			d4.z = dd;
			dd = d2[i - 2] - c2[i - 2] * dd;
			d4.y = dd;
			dd = d2[i - 3] - c2[i - 3] * dd;
			d4.x = dd;

			d_d[indg] = d4;
		}
	}
}


__global__ void GPU_adi_x_float4_2(int NX, int NY, int NZ,
	const float4* __restrict__ d_a,
	const float4* __restrict__ d_b,
	const float4* __restrict__ d_c,
	float4* __restrict__ d_d) {
	int    i, j, k, indg;
	float  aa, bb, cc, dd, c2[256], d2[256];
	float4 a4, b4, c4, d4, a4_2, b4_2, c4_2, d4_2;

	//
	// set up indices for main block
	//

	j = threadIdx.x + blockIdx.x*blockDim.x;
	k = threadIdx.y + blockIdx.y*blockDim.y;
	indg = NX*(j + k*NY) / 4;

	if ((j<NY) && (k<NZ)) {

		//
		// forward pass
		//

		a4 = d_a[indg];
		a4_2 = d_a[indg + 1];

		b4 = d_b[indg];
		b4_2 = d_b[indg + 1];

		c4 = d_c[indg];
		c4_2 = d_c[indg + 1];

		d4 = d_d[indg];
		d4_2 = d_d[indg + 1];

		bb = 1.0f / b4.x;
		cc = bb   * c4.x;
		dd = bb   * d4.x;
		c2[0] = cc;
		d2[0] = dd;

		aa = a4.y;
		bb = b4.y - aa*cc;
		dd = d4.y - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4.y;
		dd = bb*dd;
		c2[1] = cc;
		d2[1] = dd;

		aa = a4.z;
		bb = b4.z - aa*cc;
		dd = d4.z - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4.z;
		dd = bb*dd;
		c2[2] = cc;
		d2[2] = dd;

		aa = a4.w;
		bb = b4.w - aa*cc;
		dd = d4.w - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4.w;
		dd = bb*dd;
		c2[3] = cc;
		d2[3] = dd;

		aa = a4_2.x;
		bb = b4_2.x - aa*cc;
		dd = d4_2.x - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4_2.x;
		dd = bb*dd;
		c2[4] = cc;
		d2[4] = dd;

		aa = a4_2.y;
		bb = b4_2.y - aa*cc;
		dd = d4_2.y - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4_2.y;
		dd = bb*dd;
		c2[5] = cc;
		d2[5] = dd;

		aa = a4_2.z;
		bb = b4_2.z - aa*cc;
		dd = d4_2.z - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4_2.z;
		dd = bb*dd;
		c2[6] = cc;
		d2[6] = dd;

		aa = a4_2.w;
		bb = b4_2.w - aa*cc;
		dd = d4_2.w - aa*dd;
		bb = 1.0f / bb;
		cc = bb*c4_2.w;
		dd = bb*dd;
		c2[7] = cc;
		d2[7] = dd;

		for (i = 8; i<NX; i += 8) {
			indg = indg + 2;

			a4 = d_a[indg];
			a4_2 = d_a[indg + 1];

			__threadfence_block();

			b4 = d_b[indg];
			b4_2 = d_b[indg + 1];

			__threadfence_block();

			c4 = d_c[indg];
			c4_2 = d_c[indg + 1];

			__threadfence_block();

			d4 = d_d[indg];
			d4_2 = d_d[indg + 1];

			aa = a4.x;
			bb = b4.x - aa*cc;
			dd = d4.x - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.x;
			dd = bb*dd;
			c2[i] = cc;
			d2[i] = dd;

			aa = a4.y;
			bb = b4.y - aa*cc;
			dd = d4.y - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.y;
			dd = bb*dd;
			c2[i + 1] = cc;
			d2[i + 1] = dd;

			aa = a4.z;
			bb = b4.z - aa*cc;
			dd = d4.z - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.z;
			dd = bb*dd;
			c2[i + 2] = cc;
			d2[i + 2] = dd;

			aa = a4.w;
			bb = b4.w - aa*cc;
			dd = d4.w - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4.w;
			dd = bb*dd;
			c2[i + 3] = cc;
			d2[i + 3] = dd;

			aa = a4_2.x;
			bb = b4_2.x - aa*cc;
			dd = d4_2.x - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4_2.x;
			dd = bb*dd;
			c2[i + 4] = cc;
			d2[i + 4] = dd;

			aa = a4_2.y;
			bb = b4_2.y - aa*cc;
			dd = d4_2.y - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4_2.y;
			dd = bb*dd;
			c2[i + 5] = cc;
			d2[i + 5] = dd;

			aa = a4_2.z;
			bb = b4_2.z - aa*cc;
			dd = d4_2.z - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4_2.z;
			dd = bb*dd;
			c2[i + 6] = cc;
			d2[i + 6] = dd;

			aa = a4_2.w;
			bb = b4_2.w - aa*cc;
			dd = d4_2.w - aa*dd;
			bb = 1.0f / bb;
			cc = bb*c4_2.w;
			dd = bb*dd;
			c2[i + 7] = cc;
			d2[i + 7] = dd;
		}

		//
		// reverse pass
		//

		d4_2.w = dd;
		dd = d2[NX - 2] - c2[NX - 2] * dd;
		d4_2.z = dd;
		dd = d2[NX - 3] - c2[NX - 3] * dd;
		d4_2.y = dd;
		dd = d2[NX - 4] - c2[NX - 4] * dd;
		d4_2.x = dd;
		dd = d2[NX - 5] - c2[NX - 5] * dd;
		d4.w = dd;
		dd = d2[NX - 6] - c2[NX - 6] * dd;
		d4.z = dd;
		dd = d2[NX - 7] - c2[NX - 7] * dd;
		d4.y = dd;
		dd = d2[NX - 8] - c2[NX - 8] * dd;
		d4.x = dd;

		d_d[indg + 1] = d4_2;
		d_d[indg] = d4;

		for (i = NX - 9; i >= 0; i -= 8) {
			indg = indg - 2;

			dd = d2[i] - c2[i] * dd;
			d4_2.w = dd;
			dd = d2[i - 1] - c2[i - 1] * dd;
			d4_2.z = dd;
			dd = d2[i - 2] - c2[i - 2] * dd;
			d4_2.y = dd;
			dd = d2[i - 3] - c2[i - 3] * dd;
			d4_2.x = dd;
			dd = d2[i - 4] - c2[i - 4] * dd;
			d4.w = dd;
			dd = d2[i - 5] - c2[i - 5] * dd;
			d4.z = dd;
			dd = d2[i - 6] - c2[i - 6] * dd;
			d4.y = dd;
			dd = d2[i - 7] - c2[i - 7] * dd;
			d4.x = dd;

			d_d[indg + 1] = d4_2;
			d_d[indg] = d4;
		}
	}
}

//
// new tri-diagonal solve in x-direction
//

__global__ void GPU_adi_x_new(int NX, int NY, int NZ,
	const float* __restrict__ d_a,
	const float* __restrict__ d_b,
	const float* __restrict__ d_c,
	float* __restrict__ d_d)
{
	int   j, k, indg, t, tm, tp, nt, shift = 0;
	float bbi;
	__shared__  float a[256], c[256], d[256];

	//
	// set up indices for main block
	//

	t = threadIdx.x;
	j = blockIdx.x;
	k = blockIdx.y;
	indg = t + NX*(j + k*NY);

	bbi = 1.0f / d_b[indg];
	a[t] = -bbi * d_a[indg];
	c[t] = -bbi * d_c[indg];
	d[t] = bbi * d_d[indg];

	// forward pass

	tm = 2 * t;
	t = tm + 1;
	tp = tm + 2;

	for (nt = blockDim.x / 2; nt>0; nt >>= 1) {
		shift++;
		__syncthreads();

		if (threadIdx.x < nt) {
			bbi = 1.0f;
			if (tm >= 0) {
				bbi -= a[t] * c[tm];
				d[t] += a[t] * d[tm];
				a[t] = a[t] * a[tm];
			}
			if (tp<NX) {
				bbi -= c[t] * a[tp];
				d[t] += c[t] * d[tp];
				c[t] = c[t] * c[tp];
			}
			bbi = 1.0f / bbi;
			d[t] *= bbi;
			a[t] *= bbi;
			c[t] *= bbi;

			tm = 2 * tm + 1;
			t = 2 * t + 1;
			tp = 2 * tp + 1;
		}
	}

	// reverse pass

	for (; shift>0; shift--) {
		nt = blockDim.x >> shift;
		__syncthreads();

		if (threadIdx.x < nt) {
			tm >>= 1;
			t >>= 1;
			tp >>= 1;
			if (tm >= 0) d[tm] += c[tm] * d[t];
			if (tp<NX) d[tp] += a[tp] * d[t];
		}
	}

	__syncthreads();
	d_d[indg] = d[threadIdx.x];
}


//
// old tri-diagonal solve in x-direction
//

__global__ void GPU_adi_x(int NX, int NY, int NZ,
	const float* __restrict__ d_a,
	const float* __restrict__ d_b,
	const float* __restrict__ d_c,
	float* __restrict__ d_d)
{
	int   i, j, k, indg;
	float aa, bb, cc, dd, c[256], d[256];

	//
	// set up indices for main block
	//

	j = threadIdx.x + blockIdx.x*blockDim.x;  // global indices
	k = threadIdx.y + blockIdx.y*blockDim.y;
	indg = NX*(j + k*NY);

	if ((j<NY) && (k<NZ)) {

		//
		// forward pass
		//

		bb = 1.0f / d_b[indg];
		cc = bb*d_c[indg];
		dd = bb*d_d[indg];
		c[0] = cc;
		d[0] = dd;

		for (i = 1; i<NX; i++) {
			indg = indg + 1;
			aa = d_a[indg];
			bb = d_b[indg] - aa*cc;
			dd = d_d[indg] - aa*dd;
			bb = 1.0f / bb;
			cc = bb*d_c[indg];
			dd = bb*dd;
			c[i] = cc;
			d[i] = dd;
		}

		//
		// reverse pass
		//

		d_d[indg] = dd;

		for (i = NX - 2; i >= 0; i--) {
			indg = indg - 1;
			dd = d[i] - c[i] * dd;
			d_d[indg] = dd;
		}
	}
}


//
// tri-diagonal solve in y-direction
//

__global__ void GPU_adi_y(int NX, int NY, int NZ,
	const float* __restrict__ d_a,
	const float* __restrict__ d_b,
	const float* __restrict__ d_c,
	float* __restrict__ d_d)
{
	int   i, j, k, indg;
	float aa, bb, cc, dd, c[256], d[256];

	//
	// set up indices for main block
	//

	i = threadIdx.x + blockIdx.x*blockDim.x;  // global indices
	k = threadIdx.y + blockIdx.y*blockDim.y;
	indg = i + k*NX*NY;

	if ((i<NX) && (k<NZ)) {

		//
		// forward pass
		//

		bb = 1.0f / d_b[indg];
		cc = bb*d_c[indg];
		dd = bb*d_d[indg];
		c[0] = cc;
		d[0] = dd;

		for (j = 1; j<NY; j++) {
			indg = indg + NX;
			aa = d_a[indg];
			bb = d_b[indg] - aa*cc;
			dd = d_d[indg] - aa*dd;
			bb = 1.0f / bb;
			cc = bb*d_c[indg];
			dd = bb*dd;
			c[j] = cc;
			d[j] = dd;
		}

		//
		// reverse pass
		//

		d_d[indg] = dd;

		for (j = NY - 2; j >= 0; j--) {
			indg = indg - NX;
			dd = d[j] - c[j] * dd;
			d_d[indg] = dd;
		}

	}
}


//
// tri-diagonal solve in z-direction, and update solution
//

__global__ void GPU_adi_z(int NX, int NY, int NZ,
	float* __restrict__ d_u,
	const float* __restrict__ d_a,
	const float* __restrict__ d_b,
	const float* __restrict__ d_c,
	const float* __restrict__ d_d)
{
	int   i, j, k, indg, off;
	float aa, bb, cc, dd, c[256], d[256];

	//
	// set up indices for main block
	//

	i = threadIdx.x + blockIdx.x*blockDim.x;  // global indices
	j = threadIdx.y + blockIdx.y*blockDim.y;
	indg = i + j*NX;
	off = NX*NY;

	if ((i<NX) && (j<NY)) {

		//
		// forward pass
		//

		bb = 1.0f / d_b[indg];
		cc = bb*d_c[indg];
		dd = bb*d_d[indg];
		c[0] = cc;
		d[0] = dd;

		for (k = 1; k<NZ; k++) {
			indg = indg + off;
			aa = d_a[indg];
			bb = d_b[indg] - aa*cc;
			dd = d_d[indg] - aa*dd;
			bb = 1.0f / bb;
			cc = bb*d_c[indg];
			dd = bb*dd;
			c[k] = cc;
			d[k] = dd;
		}

		//
		// reverse pass
		//

		d_u[indg] += dd;

		for (k = NZ - 2; k >= 0; k--) {
			indg = indg - off;
			dd = d[k] - c[k] * dd;
			d_u[indg] += dd;
		}

	}
}




////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void Gold_adi(int, int, int, float, float*, float*, float*, float*,
	float*, float*, float*, float*, float*, float*, float*);

clock_t elapsed_time(clock_t *timer)
{
	return clock() - *timer;
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

	// 'h_' prefix - CPU (host) memory space

	int    NX = 256, NY = 256, NZ = 256, REPEAT = 10, i, j, k, ind, printout = 0;
	float  *h_u1, *h_u2, *h_du,
		*h_ax, *h_bx, *h_cx,
		*h_ay, *h_by, *h_cy,
		*h_az, *h_bz, *h_cz,
		err, lam = 1.0f;

	clock_t timer;
	double elapsed;

	// 'd_' prefix - GPU (device) memory space

	float  *d_u, *d_du,
		*d_ax, *d_bx, *d_cx,
		*d_ay, *d_by, *d_cy,
		*d_az, *d_bz, *d_cz;


	printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);

	if (NX>256 || NY>256 || NZ>256) {
		printf("No dimension can exceed 256 due to hard-coded array sizes\n");
		return -1;
	}

	// initialise card

	cutilDeviceInit(argc, argv);

	// allocate memory for arrays

	h_u1 = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_u2 = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_du = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_ax = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_bx = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_cx = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_ay = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_by = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_cy = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_az = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_bz = (float *)malloc(sizeof(float)*NX*NY*NZ);
	h_cz = (float *)malloc(sizeof(float)*NX*NY*NZ);

	cudaSafeCall(cudaMalloc((void **)&d_u, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_du, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_ax, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_bx, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_cx, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_ay, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_by, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_cy, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_az, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_bz, sizeof(float)*NX*NY*NZ));
	cudaSafeCall(cudaMalloc((void **)&d_cz, sizeof(float)*NX*NY*NZ));

	// initialise u1

	for (k = 0; k<NZ; k++) {
		for (j = 0; j<NY; j++) {
			for (i = 0; i<NX; i++) {
				ind = i + j*NX + k*NX*NY;

				if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
					h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
				else
					h_u1[ind] = 0.0f;
			}
		}
	}

	// copy u1 to device

	timer = clock();
	cudaSafeCall(cudaMemcpy(d_u, h_u1, sizeof(float)*NX*NY*NZ,
		cudaMemcpyHostToDevice));
	cudaSafeCall(cudaDeviceSynchronize());
	elapsed = elapsed_time(&timer);
	printf("\nCopy u1 to device: %f (ms) \n", elapsed);

	// Set up the execution configuration

	dim3 dimGrid1(1 + (NX - 1) / BLOCK_X, 1 + (NY - 1) / BLOCK_Y);
	dim3 dimBlock1(BLOCK_X, BLOCK_Y);

	//  dim3 dimGrid2(1+(NX-1)/16, 1+(NY-1)/4);
	//  dim3 dimBlock2(16,4);

	dim3 dimGrid2(1 + (NX - 1) / 32, 1 + (NY - 1) / 4);
	dim3 dimBlock2(32, 4);

	dim3 dimGrid3(NY, NZ);
	dim3 dimBlock3(256);

	// Execute GPU kernel

	double time1 = 0, time2 = 0, time3 = 0, time4 = 0;

	for (i = 1; i <= REPEAT; ++i) {
		GPU_adi_rhs << <dimGrid1, dimBlock1 >> >(NX, NY, NZ,
			lam, d_u, d_du,
			d_ax, d_bx, d_cx,
			d_ay, d_by, d_cy,
			d_az, d_bz, d_cz);
		cudaCheckMsg("GPU_adi_rhs execution failed\n");

		cudaSafeCall(cudaDeviceSynchronize());
		time1 += elapsed_time(&timer);

		GPU_adi_x_float4_2 << <dimGrid1, dimBlock1 >> >(NX, NY, NZ,
			(float4*)d_ax, (float4*)d_bx, (float4*)d_cx, (float4*)d_du);

		//  GPU_adi_x<<<dimGrid2, dimBlock2>>>(NX, NY, NZ,
		//                                     d_ax, d_bx, d_cx, d_du);
		//    GPU_adi_x_new<<<dimGrid3, dimBlock3>>>(NX, NY, NZ,
		//                                       d_ax, d_bx, d_cx, d_du);
		cudaCheckMsg("GPU_adi_x execution failed\n");

		cudaSafeCall(cudaDeviceSynchronize());
		time2 += elapsed_time(&timer);

		GPU_adi_y << <dimGrid2, dimBlock2 >> >(NX, NY, NZ,
			d_ay, d_by, d_cy, d_du);
		cudaCheckMsg("GPU_adi_y execution failed\n");

		cudaSafeCall(cudaDeviceSynchronize());
		time3 += elapsed_time(&timer);

		GPU_adi_z << <dimGrid2, dimBlock2 >> >(NX, NY, NZ, d_u,
			d_az, d_bz, d_cz, d_du);
		cudaCheckMsg("GPUadi_z execution failed\n");

		cudaSafeCall(cudaDeviceSynchronize());
		time4 += elapsed_time(&timer);

	}

	printf("\n%dx GPU_adi: %f (ms) %f (ms) %f (ms) %f (ms) \n",
		REPEAT, time1, time2, time3, time4);

	// Read back GPU results

	cudaSafeCall(cudaMemcpy(h_u2, d_u, sizeof(float)*NX*NY*NZ,
		cudaMemcpyDeviceToHost));
	elapsed = elapsed_time(&timer);
	printf("\nCopy u2 to host: %f (ms) \n", elapsed);

	// print out corner of array

	if (printout) {
		for (k = 0; k<3; k++) {
			for (j = 0; j<8; j++) {
				for (i = 0; i<8; i++) {
					ind = i + j*NX + k*NX*NY;
					printf(" %5.2f ", h_u2[ind]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}

	// Gold treatment

	for (int i = 1; i <= REPEAT; ++i) {
		Gold_adi(NX, NY, NZ,
			lam, h_u1, h_du,
			h_ax, h_bx, h_cx,
			h_ay, h_by, h_cy,
			h_az, h_bz, h_cz);
	}

	elapsed = elapsed_time(&timer);
	printf("\n%dx Gold_adi: %f (ms) \n \n", REPEAT, elapsed);

	// print out corner of array

	if (printout) {
		for (k = 0; k<3; k++) {
			for (j = 0; j<8; j++) {
				for (i = 0; i<8; i++) {
					ind = i + j*NX + k*NX*NY;
					printf(" %5.2f ", h_u1[ind]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}

	// error check

	err = 0.0;

	for (k = 0; k<NZ; k++) {
		for (j = 0; j<NY; j++) {
			for (i = 0; i<NX; i++) {
				ind = i + j*NX + k*NX*NY;
				err += (h_u1[ind] - h_u2[ind])*(h_u1[ind] - h_u2[ind]);
			}
		}
	}

	printf("\nrms error = %f \n", sqrt(err / (float)(NX*NY*NZ)));

	// Release GPU and CPU memory

	cudaSafeCall(cudaFree(d_u));
	cudaSafeCall(cudaFree(d_du));
	cudaSafeCall(cudaFree(d_ax));
	cudaSafeCall(cudaFree(d_bx));
	cudaSafeCall(cudaFree(d_cx));
	cudaSafeCall(cudaFree(d_ay));
	cudaSafeCall(cudaFree(d_by));
	cudaSafeCall(cudaFree(d_cy));
	cudaSafeCall(cudaFree(d_az));
	cudaSafeCall(cudaFree(d_bz));
	cudaSafeCall(cudaFree(d_cz));
	free(h_u1);
	free(h_u2);
	free(h_du);
	free(h_ax);
	free(h_bx);
	free(h_cx);
	free(h_ay);
	free(h_by);
	free(h_cy);
	free(h_az);
	free(h_bz);
	free(h_cz);

	cudaDeviceReset();

	system("pause");
}
