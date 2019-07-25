/*
 * cudaFunctions.h
 *
 *  Created on: Jan 23, 2019
 *      Author: jbohacek
 */

#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

#include "cudacoeff.h"

__device__ int ijk2n(int i, int j, int k) { 	// get linear index fot tricubic interpolation (Lekien, Marsden)
  return(i+4*j+16*k);
}

__device__ void neighborIDs(int *p, int px0, int py0, int pz0, int nx, int ny) {
	int px1 = px0+1;
	int py1 = py0+1;
	int pz1 = pz0+1;
	p[0] = px0 + nx*py0 + nx*ny*pz0;
	p[1] = px1 + nx*py0 + nx*ny*pz0;
	p[2] = px0 + nx*py1 + nx*ny*pz0;
	p[3] = px1 + nx*py1 + nx*ny*pz0;
	p[4] = px0 + nx*py0 + nx*ny*pz1;
	p[5] = px1 + nx*py0 + nx*ny*pz1;
	p[6] = px0 + nx*py1 + nx*ny*pz1;
	p[7] = px1 + nx*py1 + nx*ny*pz1;
}

template <class T>
__device__ T tricubicEval(T *a, T x, T y, T z) {
	x -= (int)x;
	y -= (int)y;
	z -= (int)z;
	T u =0.0;
	for (int i=0;i<4;i++) {
		for (int j=0;j<4;j++) {
		  for (int k=0;k<4;k++) {
		u += a[ijk2n(i,j,k)] * pow(x,i) * pow(y,j) * pow(z,k);
		  }
		}
	  }
	return(u);
}

__device__ void tricubicCoeffs(T *a, int *p,
		T *u, T *dudx, T *dudy, T *dudz, T *dudxdy, T *dudxdz, T *dudydz, T *dudxdydz) {
	for (int i=0;i<64;i++) {
	a[i]=0.0;
		for (int j=0;j<8;j++) {
			a[i]+=d_A[i][0 +j]*u[p[j]];
			a[i]+=d_A[i][8 +j]*dudx[p[j]];
			a[i]+=d_A[i][16+j]*dudy[p[j]];
			a[i]+=d_A[i][24+j]*dudz[p[j]];
			a[i]+=d_A[i][32+j]*dudxdy[p[j]];
			a[i]+=d_A[i][40+j]*dudxdz[p[j]];
			a[i]+=d_A[i][48+j]*dudydz[p[j]];
			a[i]+=d_A[i][56+j]*dudxdydz[p[j]];
		}
	}
}

// initialize CUDA fields
template <class T>
void cudaInit(T *&dux, T *&duy, T *&duz, T *&dp, T *&dm, T *&duxo, T *&duyo, T *&duzo, T *&dpo,
		      T *&duxhalf, T *&duyhalf, T *&duzhalf,												// Temperton & Staniforth, Two time level scheme
		      T *&duxc, T *&duxf, T *&duxs, T *&duxw, T *&dkuxc, T *&dkuxf, T *&dkuxs, T *&dkuxw, 	// Aux
		      T *&drx, T *&dqx, T *&dzx, T *&dpx,													// Aux
		      T *&duyc, T *&duyf, T *&duys, T *&duyw, T *&dkuyc, T *&dkuyf, T *&dkuys, T *&dkuyw, 	// Auy
		      T *&dry, T *&dqy, T *&dzy, T *&dpy,													// Auy
		      T *&duzc, T *&duzf, T *&duzs, T *&duzw, T *&dkuzc, T *&dkuzf, T *&dkuzs, T *&dkuzw, 	// Auz
		      T *&drz, T *&dqz, T *&dzz, T *&dpz,													// Auz
		      T *&dpc, T *&dpf, T *&dps, T *&dpw, T *&dkpc, T *&dkpf, T *&dkps, T *&dkpw, 			// Ap
		      T *&drp, T *&dqp, T *&dzp, T *&dpp,													// Ap
		      T *&drh, T *&dsg)
{
	cudaMemcpyToSymbol(d_dims,    &dims   ,sizeof(Dimensions));
	cudaMemcpyToSymbol(d_params,  &params ,sizeof(Parameters));
	cudaMemcpyToSymbol(d_paramsZ, &paramsZ,sizeof(ParametersZ));
	cudaMemcpyToSymbol(d_liquid, &liquid  ,sizeof(MaterialProperties));
	cudaMemcpyToSymbol(d_A, A, sizeof(A) );	// coefficient matrix for tricubic interpolation

	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;
	int blocks = params.blocks;

	cudaMalloc((void**)&dux , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// Aux
	cudaMalloc((void**)&duxhalf , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)	);
	cudaMalloc((void**)&duxo, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// old values
	cudaMalloc((void**)&drx,  sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMalloc((void**)&dqx,  sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMalloc((void**)&dzx,  sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMalloc((void**)&dpx,  sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMalloc((void**)&duxc, sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMalloc((void**)&duxf, sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)*Ny));
	cudaMalloc((void**)&duxs, sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)   ));
	cudaMalloc((void**)&duxw, sizeof(T)*((Nx-1)*Ny*Nz + 1          ));
	cudaMalloc((void**)&dkuxc,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMalloc((void**)&dkuxf,sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)*Ny));
	cudaMalloc((void**)&dkuxs,sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)   ));
	cudaMalloc((void**)&dkuxw,sizeof(T)*((Nx-1)*Ny*Nz + 1          ));

	cudaMalloc((void**)&duy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// Auy
	cudaMalloc((void**)&duyhalf, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);
	cudaMalloc((void**)&duyo, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// old values
	cudaMalloc((void**)&dry,  sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMalloc((void**)&dqy,  sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMalloc((void**)&dzy,  sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMalloc((void**)&dpy,  sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMalloc((void**)&duyc, sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMalloc((void**)&duyf, sizeof(T)*((Ny-1)*Nx*Nz +   Nx*(Ny-1)));
	cudaMalloc((void**)&duys, sizeof(T)*((Ny-1)*Nx*Nz +   Nx	   ));
	cudaMalloc((void**)&duyw, sizeof(T)*((Ny-1)*Nx*Nz + 1          ));
	cudaMalloc((void**)&dkuyc,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMalloc((void**)&dkuyf,sizeof(T)*((Ny-1)*Nx*Nz +   Nx*(Ny-1)));
	cudaMalloc((void**)&dkuys,sizeof(T)*((Ny-1)*Nx*Nz +   Nx       ));
	cudaMalloc((void**)&dkuyw,sizeof(T)*((Ny-1)*Nx*Nz + 1          ));

	cudaMalloc((void**)&duz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// Auz
	cudaMalloc((void**)&duzhalf, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);
	cudaMalloc((void**)&duzo, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// old values
	cudaMalloc((void**)&drz,  sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny	   ));
	cudaMalloc((void**)&dqz,  sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMalloc((void**)&dzz,  sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMalloc((void**)&dpz,  sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMalloc((void**)&duzc, sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMalloc((void**)&duzf, sizeof(T)*((Nz-1)*Nx*Ny +   Nx*Ny    ));
	cudaMalloc((void**)&duzs, sizeof(T)*((Nz-1)*Nx*Ny +   Nx	   ));
	cudaMalloc((void**)&duzw, sizeof(T)*((Nz-1)*Nx*Ny + 1          ));
	cudaMalloc((void**)&dkuzc,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMalloc((void**)&dkuzf,sizeof(T)*((Nz-1)*Nx*Ny +   Nx*Ny    ));
	cudaMalloc((void**)&dkuzs,sizeof(T)*((Nz-1)*Nx*Ny +   Nx       ));
	cudaMalloc((void**)&dkuzw,sizeof(T)*((Nz-1)*Nx*Ny + 1          ));

	cudaMalloc((void**)&dp , sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));  		// Aup
	cudaMalloc((void**)&dpo, sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&drp, sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&dqp, sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&dzp, sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&dpp, sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&dpc, sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&dpf, sizeof(T)*(Nx*Ny*Nz +   Nx*Ny));
	cudaMalloc((void**)&dps, sizeof(T)*(Nx*Ny*Nz +   Nx	  ));
	cudaMalloc((void**)&dpw, sizeof(T)*(Nx*Ny*Nz +    1   ));
	cudaMalloc((void**)&dkpc,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMalloc((void**)&dkpf,sizeof(T)*(Nx*Ny*Nz +   Nx*Ny));
	cudaMalloc((void**)&dkps,sizeof(T)*(Nx*Ny*Nz +   Nx   ));
	cudaMalloc((void**)&dkpw,sizeof(T)*(Nx*Ny*Nz +    1   ));

	cudaMalloc((void**)&drh,  sizeof(T)*blocks      	   );
	cudaMalloc((void**)&dsg,  sizeof(T)*blocks             );

	cudaMalloc((void**)&dm , sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));

	cudaMemset(dux  ,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)   	 ); 	// Aux
	cudaMemset(duxhalf,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)   	 );
	cudaMemset(duxo ,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)        );
	cudaMemset(drx  ,0,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMemset(dqx  ,0,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMemset(dzx  ,0,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMemset(dpx  ,0,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMemset(duxc ,0,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMemset(duxf ,0,sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)*Ny));
	cudaMemset(duxs ,0,sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)   ));
	cudaMemset(duxw ,0,sizeof(T)*((Nx-1)*Ny*Nz + 1          ));
	cudaMemset(dkuxc,0,sizeof(T)*((Nx-1)*Ny*Nz + 2*(Nx-1)*Ny));
	cudaMemset(dkuxf,0,sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)*Ny));
	cudaMemset(dkuxs,0,sizeof(T)*((Nx-1)*Ny*Nz +   (Nx-1)   ));
	cudaMemset(dkuxw,0,sizeof(T)*((Nx-1)*Ny*Nz + 1          ));

	cudaMemset(duy  ,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		 );		// Auy
	cudaMemset(duyhalf,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		 );
	cudaMemset(duyo ,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		 );		// old values
	cudaMemset(dry  ,0,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMemset(dqy  ,0,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMemset(dzy  ,0,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMemset(dpy  ,0,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMemset(duyc ,0,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMemset(duyf ,0,sizeof(T)*((Ny-1)*Nx*Nz +   Nx*(Ny-1)));
	cudaMemset(duys ,0,sizeof(T)*((Ny-1)*Nx*Nz +   Nx	    ));
	cudaMemset(duyw ,0,sizeof(T)*((Ny-1)*Nx*Nz + 1          ));
	cudaMemset(dkuyc,0,sizeof(T)*((Ny-1)*Nx*Nz + 2*Nx*(Ny-1)));
	cudaMemset(dkuyf,0,sizeof(T)*((Ny-1)*Nx*Nz +   Nx*(Ny-1)));
	cudaMemset(dkuys,0,sizeof(T)*((Ny-1)*Nx*Nz +   Nx       ));
	cudaMemset(dkuyw,0,sizeof(T)*((Ny-1)*Nx*Nz + 1          ));

	cudaMemset(duz  ,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// Auz
	cudaMemset(duzhalf,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);
	cudaMemset(duzo ,0,sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2)		);		// old values
	cudaMemset(drz  ,0,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny	));
	cudaMemset(dqz  ,0,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMemset(dzz  ,0,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMemset(dpz  ,0,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMemset(duzc ,0,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMemset(duzf ,0,sizeof(T)*((Nz-1)*Nx*Ny +   Nx*Ny    ));
	cudaMemset(duzs ,0,sizeof(T)*((Nz-1)*Nx*Ny +   Nx	    ));
	cudaMemset(duzw ,0,sizeof(T)*((Nz-1)*Nx*Ny + 1          ));
	cudaMemset(dkuzc,0,sizeof(T)*((Nz-1)*Nx*Ny + 2*Nx*Ny    ));
	cudaMemset(dkuzf,0,sizeof(T)*((Nz-1)*Nx*Ny +   Nx*Ny    ));
	cudaMemset(dkuzs,0,sizeof(T)*((Nz-1)*Nx*Ny +   Nx       ));
	cudaMemset(dkuzw,0,sizeof(T)*((Nz-1)*Nx*Ny + 1          ));

	cudaMemset(dp , 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));  		    // Aup
	cudaMemset(dpo, 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(drp, 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(dqp, 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(dzp, 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(dpp, 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(dpc, 0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(dpf, 0,sizeof(T)*(Nx*Ny*Nz +   Nx*Ny));
	cudaMemset(dps, 0,sizeof(T)*(Nx*Ny*Nz +   Nx   ));
	cudaMemset(dpw, 0,sizeof(T)*(Nx*Ny*Nz +    1   ));
	cudaMemset(dkpc,0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));
	cudaMemset(dkpf,0,sizeof(T)*(Nx*Ny*Nz +   Nx*Ny));
	cudaMemset(dkps,0,sizeof(T)*(Nx*Ny*Nz +   Nx   ));
	cudaMemset(dkpw,0,sizeof(T)*(Nx*Ny*Nz +    1   ));

    cudaMemset(drh,  0,sizeof(T)*blocks      		);
    cudaMemset(dsg,  0,sizeof(T)*blocks             );

    cudaMemset(dm ,  0,sizeof(T)*(Nx*Ny*Nz + 2*Nx*Ny));

}


template <class T>
void cudaInitTricubicDerivatives(T *&duxdx, T *&duxdy, T *&duxdz,
		T *&duxdxdy, T *&duxdxdz, T *&duxdydz, T *&duxdxdydz,
		T *&duydx, T *&duydy, T *&duydz,
		T *&duydxdy, T *&duydxdz, T *&duydydz, T *&duydxdydz,
		T *&duzdx, T *&duzdy, T *&duzdz,
		T *&duzdxdy, T *&duzdxdz, T *&duzdydz, T *&duzdxdydz,
		T *&x_ux, T *&y_ux, T *&z_ux,
		T *&x_uy, T *&y_uy, T *&z_uy,
		T *&x_uz, T *&y_uz, T *&z_uz)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;

	cudaMalloc((void**)&x_ux , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&y_ux , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&z_ux , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&x_uy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&y_uy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&z_uy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&x_uz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&y_uz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&z_uz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMalloc((void**)&duxdx , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duxdy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duxdz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duxdxdy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duxdxdz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duxdydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duxdxdydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMalloc((void**)&duydx , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duydy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duydxdy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duydxdz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duydydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duydxdydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMalloc((void**)&duzdx , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duzdy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duzdz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duzdxdy , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duzdxdz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duzdydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMalloc((void**)&duzdxdydz , sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMemset(x_ux, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(y_ux, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(z_ux, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(x_uy, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(y_uy, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(z_uy, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(x_uz, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(y_uz, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(z_uz, 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMemset(duxdx , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duxdy , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duxdz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duxdxdy , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duxdxdz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duxdydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duxdxdydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMemset(duydx , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duydy , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duydxdy , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duydxdz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duydydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duydxdydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));

	cudaMemset(duzdx , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duzdy , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duzdz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duzdxdy , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duzdxdz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duzdydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
	cudaMemset(duzdxdydz , 0, sizeof(T)*(Nx+2)*(Ny+2)*(Nz+2));
}

void cudaFinalizeTricubicDerivatives(T *&duxdx, T *&duxdy, T *&duxdz,
		T *&duxdxdy, T *&duxdxdz, T *&duxdydz, T *&duxdxdydz,
		T *&duydx, T *&duydy, T *&duydz,
		T *&duydxdy, T *&duydxdz, T *&duydydz, T *&duydxdydz,
		T *&duzdx, T *&duzdy, T *&duzdz,
		T *&duzdxdy, T *&duzdxdz, T *&duzdydz, T *&duzdxdydz,
		T *&x_ux, T *&y_ux, T *&z_ux,
		T *&x_uy, T *&y_uy, T *&z_uy,
		T *&x_uz, T *&y_uz, T *&z_uz)
{
	cudaFree(duxdx);
	cudaFree(duxdy);
	cudaFree(duxdz);
	cudaFree(duxdxdy);
	cudaFree(duxdxdz);
	cudaFree(duxdydz);
	cudaFree(duxdxdydz);

	cudaFree(duydx);
	cudaFree(duydy);
	cudaFree(duydz);
	cudaFree(duydxdy);
	cudaFree(duydxdz);
	cudaFree(duydydz);
	cudaFree(duydxdydz);

	cudaFree(duzdx);
	cudaFree(duzdy);
	cudaFree(duzdz);
	cudaFree(duzdxdy);
	cudaFree(duzdxdz);
	cudaFree(duzdydz);
	cudaFree(duzdxdydz);

	cudaFree(x_ux);
	cudaFree(y_ux);
	cudaFree(z_ux);
	cudaFree(x_uy);
	cudaFree(y_uy);
	cudaFree(z_uy);
	cudaFree(x_uz);
	cudaFree(y_uz);
	cudaFree(z_uz);
}

// destroy CUDA fields
void cudaFinalize(T *&dux, T *&duy, T *&duz, T *&dp, T *&dm, T *&duxo, T *&duyo, T *&duzo, T *&dpo,
		          T *&duxhalf, T *&duyhalf, T *&duzhalf,												// Temperton & Staniforth, Two time level scheme
				  T *&duxc, T *&duxf, T *&duxs, T *&duxw, T *&dkuxc, T *&dkuxf, T *&dkuxs, T *&dkuxw, 	// Aux
				  T *&drx, T *&dqx, T *&dzx, T *&dpx,													// Aux
				  T *&duyc, T *&duyf, T *&duys, T *&duyw, T *&dkuyc, T *&dkuyf, T *&dkuys, T *&dkuyw, 	// Auy
				  T *&dry, T *&dqy, T *&dzy, T *&dpy,													// Auy
				  T *&duzc, T *&duzf, T *&duzs, T *&duzw, T *&dkuzc, T *&dkuzf, T *&dkuzs, T *&dkuzw, 	// Auz
				  T *&drz, T *&dqz, T *&dzz, T *&dpz,													// Auz
				  T *&dpc, T *&dpf, T *&dps, T *&dpw, T *&dkpc, T *&dkpf, T *&dkps, T *&dkpw, 			// Ap
				  T *&drp, T *&dqp, T *&dzp, T *&dpp,													// Ap
				  T *&drh, T *&dsg)
{
	cudaFree(dux);
	cudaFree(duxhalf);
	cudaFree(duxo);
	cudaFree(duxc);	// Aux
	cudaFree(duxf);
	cudaFree(duxs);
	cudaFree(duxw);
	cudaFree(dkuxc);
	cudaFree(dkuxf);
	cudaFree(dkuxs);
	cudaFree(dkuxw);
	cudaFree(drx);
	cudaFree(dqx);
	cudaFree(dzx);
	cudaFree(dpx);

	cudaFree(duy);
	cudaFree(duyhalf);
	cudaFree(duyo);
	cudaFree(duyc);	// Auy
	cudaFree(duyf);
	cudaFree(duys);
	cudaFree(duyw);
	cudaFree(dkuyc);
	cudaFree(dkuyf);
	cudaFree(dkuys);
	cudaFree(dkuyw);
	cudaFree(dry);
	cudaFree(dqy);
	cudaFree(dzy);
	cudaFree(dpy);

	cudaFree(duz);
	cudaFree(duzhalf);
	cudaFree(duzo);
	cudaFree(duzc);	// Auz
	cudaFree(duzf);
	cudaFree(duzs);
	cudaFree(duzw);
	cudaFree(dkuzc);
	cudaFree(dkuzf);
	cudaFree(dkuzs);
	cudaFree(dkuzw);
	cudaFree(drz);
	cudaFree(dqz);
	cudaFree(dzz);
	cudaFree(dpz);

	cudaFree(dp); 	// Aup
	cudaFree(dpc);
	cudaFree(dpf);
	cudaFree(dps);
	cudaFree(dpw);
	cudaFree(dkpc);
	cudaFree(dkpf);
	cudaFree(dkps);
	cudaFree(dkpw);
	cudaFree(drp);
	cudaFree(dqp);
	cudaFree(dzp);
	cudaFree(dpp);

	cudaFree(dm);

	cudaFree(drh);
	cudaFree(dsg);

	cudaDeviceReset();
}

// patch anything to dux
template <class T>
__global__ void patchDux(T *dux)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int Nz = d_dims.Nz;
	int px =  i % (Nx+2);
	int py = (i / (Nx+2)) % (Ny+2);
	int pz =  i /((Nx+2)  * (Ny+2));

	if (i<(Nx+2)*(Ny+2)*(Nz+2)) {
		if ((px>150) && (px<175) && (py>35) && (py<45) && (pz>15) && (pz<25)) {
			dux[i] = 1.5;
		}
	}
}

// patch anything to duy
template <class T>
__global__ void patchDuy(T *duy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int Nz = d_dims.Nz;
	int px =  i % (Nx+2);
	int py = (i / (Nx+2)) % (Ny+2);
	int pz =  i /((Nx+2)  * (Ny+2));

	if (i<(Nx+2)*(Ny+2)*(Nz+2)) {
		if ((px>200) && (px<230) && (py>25) && (py<45) && (pz>25) && (pz<35)) {
			duy[i] = -1.5;
		}
	}
}

// patch anything to duy
template <class T>
__global__ void patchDuz(T *duz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int Nz = d_dims.Nz;
	int px =  i % (Nx+2);
	int py = (i / (Nx+2)) % (Ny+2);
	int pz =  i /((Nx+2)  * (Ny+2));

	if (i<(Nx+2)*(Ny+2)*(Nz+2)) {
		if ((px>50) && (px<75) && (py>15) && (py<35) && (pz>40) && (pz<60)) {
			duz[i] = -1.5;
		}
	}
}

template <class T>
__global__ void getTricubicDerivatives(T *duxdx, T *duxdy, T *duxdz,	// OK
		T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
		T *duydx, T *duydy, T *duydz,
		T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
		T *duzdx, T *duzdy, T *duzdz,
		T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz,
		const T *dux, const T *duy, const T *duz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);

	// ux derivatives
	if ((px>0) && (py>0) && (pz>0) && (px<nx-2) && (py<ny-1) && (pz<nz-1)) { // skip boundary
		duxdx[i]     = 0.5   * (dux[i+1] - dux[i-1]);
		duxdy[i]     = 0.5   * (dux[i+nx] - dux[i-nx]);
		duxdz[i]     = 0.5   * (dux[i+nx*ny] - dux[i-nx*ny]);
		duxdxdy[i]   = 0.25  * (dux[i+1+nx]     + dux[i-1-nx]     - dux[i-1+nx]     - dux[i+1-nx]    );
		duxdxdz[i]   = 0.25  * (dux[i+1+nx*ny]  + dux[i-1-nx*ny]  - dux[i-1+nx*ny]  - dux[i+1-nx*ny] );
		duxdydz[i]   = 0.25  * (dux[i+nx+nx*ny] + dux[i-nx-nx*ny] - dux[i-nx+nx*ny] - dux[i+nx-nx*ny]);
		duxdxdydz[i] = 0.125 * (dux[i+1+nx+nx*ny] + dux[i-1-nx+nx*ny] + dux[i+1-nx-nx*ny] + dux[i-1+nx-nx*ny]
							  - dux[i+1-nx+nx*ny] - dux[i-1+nx+nx*ny] - dux[i+1+nx-nx*ny] - dux[i-1-nx-nx*ny]);
	}
	// uy derivatives
	if ((px>0) && (py>0) && (pz>0) && (px<nx-1) && (py<ny-2) && (pz<nz-1)) { // skip boundary
		duydx[i]     = 0.5   * (duy[i+1] - duy[i-1]);
		duydy[i]     = 0.5   * (duy[i+nx] - duy[i-nx]);
		duydz[i]     = 0.5   * (duy[i+nx*ny] - duy[i-nx*ny]);
		duydxdy[i]   = 0.25  * (duy[i+1+nx]     + duy[i-1-nx]     - duy[i-1+nx]     - duy[i+1-nx]    );
		duydxdz[i]   = 0.25  * (duy[i+1+nx*ny]  + duy[i-1-nx*ny]  - duy[i-1+nx*ny]  - duy[i+1-nx*ny] );
		duydydz[i]   = 0.25  * (duy[i+nx+nx*ny] + duy[i-nx-nx*ny] - duy[i-nx+nx*ny] - duy[i+nx-nx*ny]);
		duydxdydz[i] = 0.125 * (duy[i+1+nx+nx*ny] + duy[i-1-nx+nx*ny] + duy[i+1-nx-nx*ny] + duy[i-1+nx-nx*ny]
							  - duy[i+1-nx+nx*ny] - duy[i-1+nx+nx*ny] - duy[i+1+nx-nx*ny] - duy[i-1-nx-nx*ny]);
	}
	// uz derivatives
	if ((px>0) && (py>0) && (pz>0) && (px<nx-1) && (py<ny-1) && (pz<nz-2)) { // skip boundary
		duzdx[i]     = 0.5   * (duz[i+1] - duz[i-1]);
		duzdy[i]     = 0.5   * (duz[i+nx] - duz[i-nx]);
		duzdz[i]     = 0.5   * (duz[i+nx*ny] - duz[i-nx*ny]);
		duzdxdy[i]   = 0.25  * (duz[i+1+nx]     + duz[i-1-nx]     - duz[i-1+nx]     - duz[i+1-nx]    );
		duzdxdz[i]   = 0.25  * (duz[i+1+nx*ny]  + duz[i-1-nx*ny]  - duz[i-1+nx*ny]  - duz[i+1-nx*ny] );
		duzdydz[i]   = 0.25  * (duz[i+nx+nx*ny] + duz[i-nx-nx*ny] - duz[i-nx+nx*ny] - duz[i+nx-nx*ny]);
		duzdxdydz[i] = 0.125 * (duz[i+1+nx+nx*ny] + duz[i-1-nx+nx*ny] + duz[i+1-nx-nx*ny] + duz[i-1+nx-nx*ny]
							  - duz[i+1-nx+nx*ny] - duz[i-1+nx+nx*ny] - duz[i+1+nx-nx*ny] - duz[i-1-nx-nx*ny]);
	}

}

// EXPLICIT reconstruction of Lagrangian trajectory
// A Semi-Lagrangian High-Order Method for Navier–Stokes Equations, Dongbin Xiu and George Em Karniadakis (2001)
// advection of ux
template <class T>
__global__ void advectUxDeparturePoint(T *x_ux, T *y_ux, T *z_ux, T *duxhalf, T *duyhalf, T *duzhalf,
		T *duxdx, T *duxdy, T *duxdz, T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
		T *duydx, T *duydy, T *duydz, T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
		T *duzdx, T *duzdy, T *duzdz, T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	T dx = d_dims.dx;
	T dt = d_params.dt;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);
	T x, y, z, ux, uy, uz;
	T a[64]; 	// coefficients of interpolant
	int p[8]; 	// Lekien's numbering


	// skip boundary values
	if ((px>0) && (py>0) && (pz>0) && (px<nx-2) && (py<ny-1) && (pz<nz-1)) { // skip boundary

		x = (T) px;
		y = (T) py;
		z = (T) pz;

		neighborIDs(p,(int)(x+0.5),(int)(y-0.5),(int)z,nx,ny);
		tricubicCoeffs(a,p,duyhalf,duydx,duydy,duydz,duydxdy,duydxdz,duydydz,duydxdydz);
		uy = tricubicEval(a,x+0.5,y-0.5,z);

		neighborIDs(p,(int)(x+0.5),(int)y,(int)(z-0.5),nx,ny);
		tricubicCoeffs(a,p,duzhalf,duzdx,duzdy,duzdz,duzdxdy,duzdxdz,duzdydz,duzdxdydz);
		uz = tricubicEval(a,x+0.5,y,z-0.5);


		x = px - 0.5 * dt * duxhalf[i] / dx;
		y = py - 0.5 * dt * uy / dx;
		z = pz - 0.5 * dt * uz / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0) 	x=0; 	if (x>nx-2)  	x=nx-2;
		if (y<0.5) 	y=0.5; 	if (y>ny-1.5) 	y=ny-1.5;
		if (z<0.5)  z=0.5;  if (z>nz-1.5)   z=nz-1.5;

		neighborIDs(p,(int)x,(int)y,(int)z,nx,ny);
		tricubicCoeffs(a,p,duxhalf,duxdx,duxdy,duxdz,duxdxdy,duxdxdz,duxdydz,duxdxdydz);
		ux = tricubicEval(a,x,y,z);

		neighborIDs(p,(int)(x+0.5),(int)(y-0.5),(int)z,nx,ny);
		tricubicCoeffs(a,p,duyhalf,duydx,duydy,duydz,duydxdy,duydxdz,duydydz,duydxdydz);
		uy = tricubicEval(a,x+0.5,y-0.5,z);

		neighborIDs(p,(int)(x+0.5),(int)y,(int)(z-0.5),nx,ny);
		tricubicCoeffs(a,p,duzhalf,duzdx,duzdy,duzdz,duzdxdy,duzdxdz,duzdydz,duzdxdydz);
		uz = tricubicEval(a,x+0.5,y,z-0.5);

		// position of midpoint (at alpha/2)
		x = px - dt * ux / dx;
		y = py - dt * uy / dx;
		z = pz - dt * uz / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0) 	x=0; 	if (x>nx-2)  	x=nx-2;
		if (y<0.5) 	y=0.5; 	if (y>ny-1.5) 	y=ny-1.5;
		if (z<0.5)  z=0.5;  if (z>nz-1.5)   z=nz-1.5;

		x_ux[i] = x;
		y_ux[i] = y;
		z_ux[i] = z;
	}
}

template <class T>
__global__ void advectUx(T *dux, T *x_ux, T *y_ux, T *z_ux, T *duxo,
		T *duxdx, T *duxdy, T *duxdz, T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
		T *duydx, T *duydy, T *duydz, T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
		T *duzdx, T *duzdy, T *duzdz, T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);
	T x, y, z;
	T a[64]; 	// coefficients of interpolant
	int p[8]; 	// Lekien's numbering

	// skip boundary values
	if ((px>0) && (py>0) && (pz>0) && (px<nx-2) && (py<ny-1) && (pz<nz-1)) { // skip boundary

		x = x_ux[i];
		y = y_ux[i];
		z = z_ux[i];

		neighborIDs(p,(int)x,(int)y,(int)z,nx,ny);
		tricubicCoeffs(a,p,duxo,duxdx,duxdy,duxdz,duxdxdy,duxdxdz,duxdydz,duxdxdydz);
		dux[i] = tricubicEval(a,x,y,z);
	}
}

// advection of uy
template <class T>
__global__ void advectUyDeparturePoint(T *x_uy, T *y_uy, T *z_uy, T *duxhalf, T *duyhalf, T *duzhalf,
				T *duxdx, T *duxdy, T *duxdz, T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
				T *duydx, T *duydy, T *duydz, T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
				T *duzdx, T *duzdy, T *duzdz, T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	T dx = d_dims.dx;
	T dt = d_params.dt;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);
	T x, y, z, ux, uy, uz;
	T a[64]; 	// coefficients of interpolant
	int p[8]; 	// Lekien's numbering

	// skip boundary values
	if ((px>0) && (py>0) && (pz>0) && (px<nx-1) && (py<ny-2) && (pz<nz-1)) { // skip boundary

		x = (T) px;
		y = (T) py;
		z = (T) pz;

		neighborIDs(p,(int)(x-0.5),(int)(y+0.5),(int)z,nx,ny);
		tricubicCoeffs(a,p,duxhalf,duxdx,duxdy,duxdz,duxdxdy,duxdxdz,duxdydz,duxdxdydz);
		ux = tricubicEval(a,x-0.5,y+0.5,z);

		neighborIDs(p,(int)x,(int)(y+0.5),(int)(z-0.5),nx,ny);
		tricubicCoeffs(a,p,duzhalf,duzdx,duzdy,duzdz,duzdxdy,duzdxdz,duzdydz,duzdxdydz);
		uz = tricubicEval(a,x,y+0.5,z-0.5);

		// move "backwards" in time
		x = px - 0.5 * dt * ux / dx;
		y = py - 0.5 * dt * duyhalf[i] / dx;
		z = pz - 0.5 * dt * uz / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0.5) 	x=0.5; 	if (x>nx-1.5)	x=nx-1.5;
		if (y<0) 	y=0; 	if (y>ny-2) 	y=ny-2;
		if (z<0.5) 	z=0.5; 	if (z>nz-1.5) 	z=nz-1.5;

		neighborIDs(p,(int)(x-0.5),(int)(y+0.5),(int)z,nx,ny);
		tricubicCoeffs(a,p,duxhalf,duxdx,duxdy,duxdz,duxdxdy,duxdxdz,duxdydz,duxdxdydz);
		ux = tricubicEval(a,x-0.5,y+0.5,z);

		neighborIDs(p,(int)x,(int)y,(int)z,nx,ny);
		tricubicCoeffs(a,p,duyhalf,duydx,duydy,duydz,duydxdy,duydxdz,duydydz,duydxdydz);
        uy = tricubicEval(a,x,y,z);

		neighborIDs(p,(int)x,(int)(y+0.5),(int)(z-0.5),nx,ny);
		tricubicCoeffs(a,p,duzhalf,duzdx,duzdy,duzdz,duzdxdy,duzdxdz,duzdydz,duzdxdydz);
		uz = tricubicEval(a,x,y+0.5,z-0.5);

		// position of midpoint (at alpha/2)
		x = px - dt * ux / dx;
		y = py - dt * uy / dx;
		z = pz - dt * uz / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0.5) 	x=0.5; 	if (x>nx-1.5)  	x=nx-1.5;
		if (y<0) 	y=0; 	if (y>ny-2) 	y=ny-2;
		if (z<0.5)  z=0.5;  if (z>nz-1.5)   z=nz-1.5;

		x_uy[i] = x;
		y_uy[i] = y;
		z_uy[i] = z;
	}
}

// advection of uy
template <class T>
__global__ void advectUy(T *duy, T *x_uy, T *y_uy, T *z_uy, T *duyo,
				T *duxdx, T *duxdy, T *duxdz, T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
				T *duydx, T *duydy, T *duydz, T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
				T *duzdx, T *duzdy, T *duzdz, T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);
	T x, y, z;
	T a[64]; 	// coefficients of interpolant
	int p[8]; 	// Lekien's numbering

	// skip boundary values
	if ((px>0) && (py>0) && (pz>0) && (px<nx-1) && (py<ny-2) && (pz<nz-1)) { // skip boundary

		x = x_uy[i];
		y = y_uy[i];
		z = z_uy[i];

		neighborIDs(p,(int)x,(int)y,(int)z,nx,ny);
		tricubicCoeffs(a,p,duyo,duydx,duydy,duydz,duydxdy,duydxdz,duydydz,duydxdydz);
        duy[i] = tricubicEval(a,x,y,z);
	}
}

// advection of uz
template <class T>
__global__ void advectUzDeparturePoint(T *x_uz, T *y_uz, T *z_uz, T *duxhalf, T *duyhalf, T *duzhalf,
				 T *duxdx, T *duxdy, T *duxdz, T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
				 T *duydx, T *duydy, T *duydz, T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
				 T *duzdx, T *duzdy, T *duzdz, T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	T dx = d_dims.dx;
	T dt = d_params.dt;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);
	T x, y, z, ux, uy, uz;
	T a[64]; 	// coefficients of interpolant
	int p[8]; 	// Lekien's numbering

	// skip boundary values
	if ((px>0) && (py>0) && (pz>0) && (px<nx-1) && (py<ny-1) && (pz<nz-2)) { // skip boundary

		x = (T) px;
		y = (T) py;
		z = (T) pz;

		neighborIDs(p,(int)(x-0.5),(int)y,(int)(z+0.5),nx,ny);
		tricubicCoeffs(a,p,duxhalf,duxdx,duxdy,duxdz,duxdxdy,duxdxdz,duxdydz,duxdxdydz);
		ux = tricubicEval(a,x-0.5,y,z+0.5);

		neighborIDs(p,(int)x,(int)(y-0.5),(int)(z+0.5),nx,ny);
		tricubicCoeffs(a,p,duyhalf,duydx,duydy,duydz,duydxdy,duydxdz,duydydz,duydxdydz);
		uy = tricubicEval(a,x,y-0.5,z+0.5);

		// move "backwards" in time
		x = px - 0.5 * dt * ux / dx;
		y = py - 0.5 * dt * uy / dx;
		z = pz - 0.5 * dt * duzhalf[i] / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0.5) 	x=0.5; 	if (x>nx-1.5)	x=nx-1.5;
		if (y<0.5) 	y=0.5; 	if (y>ny-1.5) 	y=ny-1.5;
		if (z<0) 	z=0; 	if (z>nz-2) 	z=nz-2;

		neighborIDs(p,(int)(x-0.5),(int)y,(int)(z+0.5),nx,ny);
		tricubicCoeffs(a,p,duxhalf,duxdx,duxdy,duxdz,duxdxdy,duxdxdz,duxdydz,duxdxdydz);
		ux = tricubicEval(a,x-0.5,y,z+0.5);

		neighborIDs(p,(int)x,(int)(y-0.5),(int)(z+0.5),nx,ny);
		tricubicCoeffs(a,p,duyhalf,duydx,duydy,duydz,duydxdy,duydxdz,duydydz,duydxdydz);
		uy = tricubicEval(a,x,y-0.5,z+0.5);

		neighborIDs(p,(int)x,(int)y,(int)z,nx,ny);
		tricubicCoeffs(a,p,duzhalf,duzdx,duzdy,duzdz,duzdxdy,duzdxdz,duzdydz,duzdxdydz);
		uz = tricubicEval(a,x,y,z);

		// position of midpoint (at alpha/2)
		x = px - dt * ux / dx;
		y = py - dt * uy / dx;
		z = pz - dt * uz / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0.5) 	x=0.5; 	if (x>nx-1.5)	x=nx-1.5;
		if (y<0.5) 	y=0.5; 	if (y>ny-1.5) 	y=ny-1.5;
		if (z<0) 	z=0; 	if (z>nz-2) 	z=nz-2;

		x_uz[i] = x;
		y_uz[i] = y;
		z_uz[i] = z;
	}
}

// advection of uz
template <class T>
__global__ void advectUz(T *duz, T *x_uz, T *y_uz, T *z_uz, T *duzo,
				 T *duxdx, T *duxdy, T *duxdz, T *duxdxdy, T *duxdxdz, T *duxdydz, T *duxdxdydz,
				 T *duydx, T *duydy, T *duydz, T *duydxdy, T *duydxdz, T *duydydz, T *duydxdydz,
				 T *duzdx, T *duzdy, T *duzdz, T *duzdxdy, T *duzdxdz, T *duzdydz, T *duzdxdydz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;
	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);
	T x, y, z;
	T a[64]; 	// coefficients of interpolant
	int p[8]; 	// Lekien's numbering

	// skip boundary values
	if ((px>0) && (py>0) && (pz>0) && (px<nx-1) && (py<ny-1) && (pz<nz-2)) { // skip boundary

		x = x_uz[i];
		y = y_uz[i];
		z = z_uz[i];

		neighborIDs(p,(int)x,(int)y,(int)z,nx,ny);
		tricubicCoeffs(a,p,duzo,duzdx,duzdy,duzdz,duzdxdy,duzdxdz,duzdydz,duzdxdydz);
 		duz[i] = tricubicEval(a,x,y,z);
	}
}

// NO slip wall for velocity
template <class T>
__global__ void bcVelWallNoslip(T *dux, T *duy, T *duz)  // 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int nz = d_dims.Nz+2;

	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i /(nx  * ny);

	// Skip Inner Values

	if (pz==0) {
		dux[i] = -dux[i+nx*ny];
		duy[i] = -duy[i+nx*ny];
		duz[i] = 0;
	}

	if (pz==nz-1) {
		dux[i] = -dux[i-nx*ny];
		duy[i] = -duy[i-nx*ny];
		duz[i] = 0;
		duz[i-nx*ny] = 0;
	}

	if (py==0) {
		dux[i] = -dux[i+nx];
		duy[i] = 0;
		duz[i] = -duz[i+nx];
	}

	if (py==ny-1) {
		dux[i] = -dux[i-nx];
		duy[i] = 0;
		duy[i-nx] = 0;
		duz[i] = -duz[i-nx];
	}

	if (px==0) {
		dux[i] = 0;
		duy[i] = -duy[i+1];
		duz[i] = -duz[i+1];
	}

	if (px==nx-1) {
		dux[i] = 0;
		dux[i-1] = 0;
		duy[i] = -duy[i-1];
		duz[i] = -duz[i-1];
	}
}


// velocity inlet (shroud)
template <class T>
__global__ void bcVelInlet(T *duz, const T UZ, const int startx, const int starty) // 1_OK
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int nz = d_dims.Nz;

	int px = i%10;
	int py = i/10;

	duz[px+startx+1 + (py+starty+1)*(nx+2) + (nx+2)*(ny+2)*nz] = UZ;
	// zero tangential velocity already implemented in bcVelWallNoSlip
}

// velocity outlet (sen)
template <class T>
__global__ void bcVelOutlet(T *dux, T *duy, T *duz, const int startx, const int starty)  // 1_OK
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;

	int px = i%10;
	int py = i/10;

	int second = px+startx+1 + (py+starty+1)*nx + nx*ny;

	duz[second-nx*ny] = duz[second]
	                  + duy[second] - duy[second-nx]
	                  + dux[second] - dux[second-1]; // fulfil continuity

	if (i>0) {
		dux[second-1 -nx*ny] = dux[second-1 ];	// zero gradient
		duy[second-nx-nx*ny] = duy[second-nx];
	}
}

// explicit part of Crank-Nicolson scheme for diffusion term for x-component of velocity
template <class T>
__global__ void expMux(T *drx,
		T *dux,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;

	T nu   = d_liquid.nu;

	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	int second = px+1+(py+1)*nx+(pz+1)*nx*ny;

	if (i<Nx*Ny*Nz)
	{
		drx[i+Nx*Ny] += -3*nu*dux[second] + 0.5*nu*(dux[second-1]     + dux[second+1]
		                                          + dux[second-nx]    + dux[second+nx]
		                                          + dux[second-nx*ny] + dux[second+nx*ny]);

		if (px==0)	  drx[i+Nx*Ny] += -0.5*nu*dux[second-1];
		if (py==0)    drx[i+Nx*Ny] += -0.5*nu*dux[second] - 0.5*nu*dux[second-nx];
		if (pz==0)    drx[i+Nx*Ny] += -0.5*nu*dux[second] - 0.5*nu*dux[second-nx*ny];
		if (px==Nx-1) drx[i+Nx*Ny] += -0.5*nu*dux[second+1];
		if (py==Ny-1) drx[i+Nx*Ny] += -0.5*nu*dux[second] - 0.5*nu*dux[second+nx];
		if (pz==Nz-1) drx[i+Nx*Ny] += -0.5*nu*dux[second] - 0.5*nu*dux[second+nx*ny];
	}
}

// explicit part of Crank-Nicolson scheme for diffusion term for y-component of velocity
template <class T>
__global__ void expMuy(T *dry,
		T *duy,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;

	T nu   = d_liquid.nu;

	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	int second = px+1+(py+1)*nx+(pz+1)*nx*ny;

	if (i<Nx*Ny*Nz)
	{
		dry[i+Nx*Ny] += -3*nu*duy[second] + 0.5*nu*(duy[second-1]     + duy[second+1]
		                                          + duy[second-nx]    + duy[second+nx]
		                                          + duy[second-nx*ny] + duy[second+nx*ny]);

		if (px==0)	  dry[i+Nx*Ny] += -0.5*nu*duy[second] - 0.5*nu*duy[second-1];
		if (py==0)    dry[i+Nx*Ny] += -0.5*nu*duy[second-nx];
		if (pz==0)    dry[i+Nx*Ny] += -0.5*nu*duy[second] - 0.5*nu*duy[second-nx*ny];
		if (px==Nx-1) dry[i+Nx*Ny] += -0.5*nu*duy[second] - 0.5*nu*duy[second+1];
		if (py==Ny-1) dry[i+Nx*Ny] += -0.5*nu*duy[second+nx];
		if (pz==Nz-1) dry[i+Nx*Ny] += -0.5*nu*duy[second] - 0.5*nu*duy[second+nx*ny];
	}
}

// explicit part of Crank-Nicolson scheme for diffusion term for z-component of velocity
template <class T>
__global__ void expMuz(T *drz,
		T *duz,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;

	T nu   = d_liquid.nu;

	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	int second = px+1+(py+1)*nx+(pz+1)*nx*ny;

	if (i<Nx*Ny*Nz)
	{
		drz[i+Nx*Ny] += -3*nu*duz[second] + 0.5*nu*(duz[second-1]     + duz[second+1]
		                                          + duz[second-nx]    + duz[second+nx]
		                                          + duz[second-nx*ny] + duz[second+nx*ny]);

		if (px==0)	  drz[i+Nx*Ny] += -0.5*nu*duz[second] - 0.5*nu*duz[second-1];
		if (py==0)    drz[i+Nx*Ny] += -0.5*nu*duz[second] - 0.5*nu*duz[second-nx];
		if (pz==0)    drz[i+Nx*Ny] += -0.5*nu*duz[second-nx*ny];
		if (px==Nx-1) drz[i+Nx*Ny] += -0.5*nu*duz[second] - 0.5*nu*duz[second+1];
		if (py==Ny-1) drz[i+Nx*Ny] += -0.5*nu*duz[second] - 0.5*nu*duz[second+nx];
		if (pz==Nz-1) drz[i+Nx*Ny] += -0.5*nu*duz[second+nx*ny];
	}
}

// fill diagonals of Aux
template <class T>
__global__ void Aux(T *duxc, T *duxf, T *duxs, T *duxw)	// launch (Nx-1)*Ny*Nz threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx-1;
	int ny = d_dims.Ny;
	int nz = d_dims.Nz;

	T ac   = d_params.ac;
	T nu   = d_liquid.nu;
	T urf  = d_params.urfU;

	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i / (nx*ny);

	if (i<nx*ny*nz)
	{
		duxc[i+nx*ny] = ac/urf + 3*nu;
		duxf[i] = -0.5*nu;
		duxs[i] = -0.5*nu;
		duxw[i] = -0.5*nu;

		if (px==0)	duxw[i] = 0;

		if (py==0) {
			duxc[i+nx*ny] += 0.5*nu;
			duxs[i] = 0;
		}

		if (pz==0) {
			duxc[i+nx*ny] += 0.5*nu;
			duxf[i] = 0;
		}

		if (py==ny-1)	duxc[i+nx*ny] += 0.5*nu;

		if (pz==nz-1)	duxc[i+nx*ny] += 0.5*nu;
	}
}

// fill diagonals of Auy
template <class T>
__global__ void Auy(T *duyc, T *duyf, T *duys, T *duyw)	// launch Nx*(Ny-1)*Nz threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny-1;
	int nz = d_dims.Nz;

	T ac   = d_params.ac;
	T nu   = d_liquid.nu;
	T urf  = d_params.urfU;

	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i / (nx*ny);

	if (i<nx*ny*nz)
	{
		duyc[i+nx*ny] = ac/urf + 3*nu;
		duyf[i] = -0.5*nu;
		duys[i] = -0.5*nu;
		duyw[i] = -0.5*nu;

		if (px==0) {
			duyc[i+nx*ny] += 0.5*nu;
			duyw[i] = 0;
		}

		if (py==0)	duys[i] = 0;

		if (pz==0) {
			duyc[i+nx*ny] += 0.5*nu;
			duyf[i] = 0;
		}

		if (px==nx-1)	duyc[i+nx*ny] += 0.5*nu;

		if (pz==nz-1)	duyc[i+nx*ny] += 0.5*nu;
	}
}


// fill diagonals of Auz
template <class T>
__global__ void Auz(T *duzc, T *duzf, T *duzs, T *duzw)	// launch Nx*Ny*(Nz-1) threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int nz = d_dims.Nz-1;

	T ac   = d_params.ac;
	T nu   = d_liquid.nu;
	T urf  = d_params.urfU;

	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i / (nx*ny);

	if (i<nx*ny*nz)
	{
		duzc[i+nx*ny] = ac/urf + 3*nu;
		duzf[i] = -0.5*nu;
		duzs[i] = -0.5*nu;
		duzw[i] = -0.5*nu;

		if (px==0) {
			duzc[i+nx*ny] += 0.5*nu;
			duzw[i] = 0;
		}

		if (py==0) {
			duzc[i+nx*ny] += 0.5*nu;
			duzs[i] = 0;
		}

		if (pz==0)	duzf[i] = 0;

		if (px==nx-1)	duzc[i+nx*ny] += 0.5*nu;

		if (py==ny-1)	duzc[i+nx*ny] += 0.5*nu;
	}
}

// fill diagonals of Ap
template <class T>
__global__ void Ap(T *dpc, T *dpf, T *dps, T *dpw)	// launch Nx*Ny*Nz threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int nz = d_dims.Nz;

	T D = d_params.dt / (d_liquid.rho * d_dims.dx);

	int px = i % nx;
	int py =(i / nx) % ny;
	int pz = i / (nx*ny);

	if (i<nx*ny*nz)
	{
		dpc[i+nx*ny] = 6*D;
		dpf[i] = -D;
		dps[i] = -D;
		dpw[i] = -D;

		if (px==0) {
			dpc[i+nx*ny] -= D;
			dpw[i] = 0;
		}

		if (py==0)	{
			dpc[i+nx*ny] -= D;
			dps[i] = 0;
		}

		if (pz==0)	{
			dpc[i+nx*ny] -= D;
			dpf[i] = 0;
		}

		if (px==nx-1)	dpc[i+nx*ny] -= D;

		if (py==ny-1)	dpc[i+nx*ny] -= D;

		if (pz==nz-1)	dpc[i+nx*ny] -= D;
	}
}


// modify diagonals of Aux at outlet
template <class T>
__global__ void AuxOutlet(T *duxc, const int startx, const int starty)	// launch outletWidth number of threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx-1;
	int ny = d_dims.Ny;

	int px = i%10;
	int py = i/10;

	T nu   = d_liquid.nu;

	if (px>0) 	duxc[startx+px-1 + (starty+py)*nx + nx*ny] -= 1*nu;  // no viscous force; zero gradient boundary condition

}

// modify diagonals of Auy at outlet
template <class T>
__global__ void AuyOutlet(T *duyc, const int startx, const int starty)	// launch outletWidth number of threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny-1;

	int px = i%10;
	int py = i/10;

	T nu   = d_liquid.nu;

	if (py>0)	duyc[startx+px + (starty+py-1)*nx + nx*ny] -= 1*nu; // no viscous force; zero gradient boundary condition
}

// modify diagonals of Auz at outlet
template <class T>
__global__ void AuzOutlet(T *duzc, const int startx, const int starty)	// launch outletWidth number of threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;

	int px = i%10;
	int py = i/10;

	T nu   = d_liquid.nu;

	duzc[startx+px + (starty+py)*nx + nx*ny] -= 0.5*nu; // no viscous force; zero gradient boundary condition
}


// modify diagonals of Ap at outlet
template <class T>
__global__ void ApOutlet(T *dpc, const int startx, const int starty)	// launch outletWidth number of threads; 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;

	int px = i%10;
	int py = i/10;

	T D = d_params.dt / (d_liquid.rho * d_dims.dx);

	dpc[startx+px + (starty+py)*nx + nx*ny]   = 6*D; // Dirichlet at outlet = zero pressure
}


// modify by at inlet (Note: at inlet bx,by without change)
template <class T>
__global__ void bzInlet(T *drz, const T *duz, const int startx, const int starty) // 1_OK
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int nz = d_dims.Nz;

	int px = i%10;
	int py = i/10;

	T nu   = d_liquid.nu;

	drz[startx+px + (starty+py)*nx + nx*ny*(nz-1)] += nu * duz[startx+px+1 + (starty+py+1)*(nx+2) + (nx+2)*(ny+2)*nz];
}

// fill bpx add dp/dx to right-handside Aux = bx
template <class T>
__global__ void bpx(T *dr,	// 1_OK
		const T *dp,
		const int Nx,
		const int Ny,
		const int Nz)	// launch (Nx-1)*Ny*Nz threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	T rho  = d_liquid.rho;
	T dx   = d_dims.dx;

	if (i<Nx*Ny*Nz)	dr[i+Nx*Ny] += dx * (dp[px+py*nx+(pz+1)*nx*ny]  -  dp[px+1+py*nx+(pz+1)*nx*ny]) / rho;  // pw-pe;
}

// fill bpy add dp/dy to right-handside of Auy = by
template <class T>
__global__ void bpy(T *dr,	// 1_OK
		const T *dp,
		const int Nx,
		const int Ny,
		const int Nz)	// launch Nx*(Ny-1)*Nz threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	T rho  = d_liquid.rho;
	T dx   = d_dims.dx;

	if (i<Nx*Ny*Nz)	dr[i+Nx*Ny] += dx * (dp[px+py*nx+(pz+1)*nx*ny]  -  dp[px+(py+1)*nx+(pz+1)*nx*ny]) / rho;  // ps-pn;
}

// fill bpy add dp/dy to right-handside of Auz = bz
template <class T>
__global__ void bpz(T *dr,	// 1_OK
		const T *dp,
		const int Nx,
		const int Ny,
		const int Nz)	// launch Nx*Ny*(Nz-1) threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	T rho  = d_liquid.rho;
	T dx   = d_dims.dx;

	if (i<Nx*Ny*Nz)	dr[i+Nx*Ny] += dx * (dp[px+py*nx+(pz+1)*nx*ny]  -  dp[px+py*nx+(pz+2)*nx*ny]) / rho;  // pf-pb;
}

// update UX field after solving P pressure
template <class T>
__global__ void correctUX(T *dux,	// 1_OK
		const T *dp,
		const int Nx,
		const int Ny,
		const int Nz)	// launch (Nx-1)*Ny*Nz threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);
	T rho  = d_liquid.rho;
	T dt   = d_params.dt;
	T dx   = d_dims.dx;

	if (i<Nx*Ny*Nz)	dux[px+1+(py+1)*(nx+2)+(pz+1)*(nx+2)*(ny+2)] -=  dt * (dp[px+1+py*nx+(pz+1)*nx*ny]  -  dp[px+py*nx+(pz+1)*nx*ny]) / (rho*dx);  // pe-pw
}

// update UY field after solving P pressure
template <class T>
__global__ void correctUY(T *duy,	// 1_OK
		const T *dp,
		const int Nx,
		const int Ny,
		const int Nz)	// launch Nx*(Ny-1)*Nz threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);
	T rho  = d_liquid.rho;
	T dt   = d_params.dt;
	T dx   = d_dims.dx;

	if (i<Nx*Ny*Nz)	duy[px+1+(py+1)*(nx+2)+(pz+1)*(nx+2)*(ny+2)] -=  dt * (dp[px+(py+1)*nx+(pz+1)*nx*ny]  -  dp[px+py*nx+(pz+1)*nx*ny]) / (rho*dx);  // pn-ps
}

// update UZ field after solving P pressure
template <class T>
__global__ void correctUZ(T *duz,	// 1_OK
		const T *dp,
		const int Nx,
		const int Ny,
		const int Nz)	// launch Nx*Ny*(Nz-1) threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);
	T rho  = d_liquid.rho;
	T dt   = d_params.dt;
	T dx   = d_dims.dx;

	if (i<Nx*Ny*Nz)	duz[px+1+(py+1)*(nx+2)+(pz+1)*(nx+2)*(ny+2)] -=  dt * (dp[px+py*nx+(pz+2)*nx*ny]  -  dp[px+py*nx+(pz+1)*nx*ny]) / (rho*dx);  // pb-pf
}


// **********************************************************
// ****** generic functions for all unknowns ux, uy, uz, p ******
// **********************************************************


// copy dux to drx
template <class T>
__global__ void duToDr(T *dr,	// 1_OK
		               const T *du,
		               const int Nx,
		               const int Ny,
		               const int Nz)
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	if (i<Nx*Ny*Nz)	dr[i+Nx*Ny] = du[px+1 + (py+1)*nx + (pz+1)*nx*ny];
}


// fill b (Ax=b); right-handside of Au = b
template <class T>
__global__ void b(T *dr,  // 1_OK
		const int Nx,
		const int Ny,
		const int Nz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	T ac   = d_params.ac;
	T urf  = d_params.urfU;

	if (i<Nx*Ny*Nz)	dr[i+Nx*Ny] *= ac/urf;
}

// fill b (Ax=b); right-handside of Au = b
template <class T>
__global__ void bp(T *dr,	//1_OK
		const T *dux,
		const T *duy,
		const T *duz,
		const int Nx,
		const int Ny,
		const int Nz)	// launch Nx*Ny*Nz threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx+2;
	int ny = d_dims.Ny+2;
	int px = i % Nx;
	int py =(i / Nx) % Ny;
	int pz = i / (Nx*Ny);

	int second = px+1 + (py+1)*nx + (pz+1)*nx*ny;

	if (i<Nx*Ny*Nz)	dr[i+Nx*Ny] = dux[second-1]     - dux[second]
			                    + duy[second-nx]    - duy[second]
			                    + duz[second-nx*ny] - duz[second];
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMV(T *y,
		const T *dc,
		const T *df,
		const T *ds,
		const T *dw,
		const T *x,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	int NxNy = Nx*Ny;
	int tids = tid + NxNy;  // tid shifted

	if (tid<NxNy*Nz) {
		y[tids] = dc[tid+NxNy]  * x[tids]        // center
				+ ds[tid+Nx]    * x[tids+Nx]     // north               N
				+ dw[tid+1]     * x[tids+1]      // east              W C E
				+ ds[tid]       * x[tids-Nx]     // south               S
				+ dw[tid]       * x[tids-1]      // west
				+ df[tid+NxNy]  * x[tids+NxNy]   // back                B
				+ df[tid]       * x[tids-NxNy];  // front               F
	}
}


// Truncated Neumann series 1 in 3D
template <class T>
__global__ void makeTNS1(T *dkc,
		T *dkf,
		T *dks,
		T *dkw,
		const T *dc,
		const T *df,
		const T *ds,
		const T *dw,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int tid    = threadIdx.x + blockIdx.x * blockDim.x;
	int NxNy   = Nx*Ny;
	int NxNyNz = Nx*Ny*Nz;

	if (tid<NxNyNz) {
		int tids   = tid + NxNy;  // tid shifted

		T tstC1 = 1. / dc[tids];
		T tstC2 = 0.;
		T tstC3 = 0.;
		T tstC5 = 0.;

		if (tid < NxNyNz-NxNy)   tstC5 = 1. / dc[tids+NxNy];    // dkf
		if (tid < NxNyNz-Nx)     tstC2 = 1. / dc[tids+Nx];      // dks
		if (tid < NxNyNz-1)      tstC3 = 1. / dc[tids+1];       // dkw

		dkc[tid+NxNy] = tstC1 * (1. + dw[tid+1]    * dw[tid+1]    * tstC1 * tstC3
									+ ds[tid+Nx]   * ds[tid+Nx]   * tstC1 * tstC2
									+ df[tid+NxNy] * df[tid+NxNy] * tstC1 * tstC5);

		dkw[tid+1]    = -dw[tid+1]  * tstC1 * tstC3;
		dks[tid+Nx]   = -ds[tid+Nx] * tstC1 * tstC2;
		dkf[tid+NxNy] = -df[tid+NxNy] * tstC1 * tstC5;
		}
}

// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPY(T *y,	// launch Nx*Ny*Nz threads
		const T *x,
		const T alpha,
		const T beta,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid<Nx*Ny*Nz) {
		tid += Nx*Ny;
		y[tid] = alpha * x[tid] + beta * y[tid];
	}
}

// AXPY2 (y := alpha*x + beta*y)
// sizeof(dy) != sizeof(x)
template <class T>
__global__ void AXPY2(T *y,	// launch Nx*Ny*Nz threads; 1_OK
		const T *x,
		const T alpha,
		const T beta,
		const int Nx,
		const int Ny,
		const int Nz)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<Nx*Ny*Nz) {
		int nx = d_dims.Nx;
		int ny = d_dims.Ny;
		int px = i % Nx;
		int py =(i / Nx) % Ny;
		int pz = i / (Nx*Ny);

		y[px+1+(py+1)*(nx+2)+(pz+1)*(nx+2)*(ny+2)] = alpha * x[i+Nx*Ny] + beta * y[px+1+(py+1)*(nx+2)+(pz+1)*(nx+2)*(ny+2)];
	}
}

// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPU(T *c,
		const T *a,
		const T *b,
		const int Nx,
		const int Ny,
		const int Nz)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;
	unsigned int NxNy = Nx*Ny;
	unsigned int NxNyNz = Nx*Ny*Nz;

	cache[tid] = 0;

	while(i<NxNyNz) {
		cache[tid] += a[i+NxNy] * b[i+NxNy] + a[i+NxNy+blockSize] * b[i+NxNy+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) c[blockIdx.x] = cache[0];
}



#endif /* CUDAFUNCTIONS_H_ */
