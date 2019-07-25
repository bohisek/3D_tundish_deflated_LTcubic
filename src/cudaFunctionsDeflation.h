#ifndef CUDAFUNCTIONSDEFLATION_H_
#define CUDAFUNCTIONSDEFLATION_H_

// initialize CUDA fields
template <class T>
void cudaInitDeflation( T *&dpzc, T *&dpzf, T *&dpzs, T *&dpzw,
		T *&dec, T *&def, T *&des, T *&dew,
		T *&drZ, T *&dyZ, T *&dqZ, T *&dpZ,
		T *&drhs,
		const T *ec, const T *ef, const T *es, const T *ew,
		const T *pzc, const T *pzf, const T *pzs, const T *pzw)
{

	int Nx  = dims.Nx;
	int Ny  = dims.Ny;
	int Nz  = dims.Nz;
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;
	int nDV = paramsZ.nDV;

	cudaMalloc((void**)&dpzc ,sizeof(T)*(Nx*Ny*Nz));
	cudaMalloc((void**)&dpzf ,sizeof(T)*(Nx*Ny*Nz+Nx*Ny));
	cudaMalloc((void**)&dpzs ,sizeof(T)*(Nx*Ny*Nz+Nx));
	cudaMalloc((void**)&dpzw ,sizeof(T)*(Nx*Ny*Nz+1));

	cudaMalloc((void**)&drhs ,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));

	cudaMalloc((void**)&dyZ  ,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMalloc((void**)&drZ  ,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMalloc((void**)&dpZ  ,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMalloc((void**)&dqZ  ,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMalloc((void**)&dec  ,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMalloc((void**)&def  ,sizeof(T)*(nDV+NxZ*NyZ));
	cudaMalloc((void**)&des  ,sizeof(T)*(nDV+NxZ));
	cudaMalloc((void**)&dew  ,sizeof(T)*(nDV+1));

	cudaMemcpy(dpzc,pzc,sizeof(T)*(Nx*Ny*Nz)      ,cudaMemcpyHostToDevice);
	cudaMemcpy(dpzf,pzf,sizeof(T)*(Nx*Ny*Nz+Nx*Ny),cudaMemcpyHostToDevice);
	cudaMemcpy(dpzs,pzs,sizeof(T)*(Nx*Ny*Nz+Nx)   ,cudaMemcpyHostToDevice);
	cudaMemcpy(dpzw,pzw,sizeof(T)*(Nx*Ny*Nz+1)    ,cudaMemcpyHostToDevice);

	cudaMemcpy(dec,ec,sizeof(T)*(nDV+2*NxZ*NyZ),cudaMemcpyHostToDevice);
	cudaMemcpy(def,ef,sizeof(T)*(nDV+NxZ*NyZ)  ,cudaMemcpyHostToDevice);
	cudaMemcpy(des,es,sizeof(T)*(nDV+NxZ)  	   ,cudaMemcpyHostToDevice);
	cudaMemcpy(dew,ew,sizeof(T)*(nDV+1)        ,cudaMemcpyHostToDevice);

	cudaMemset(drhs,0,sizeof(T)*(Nx*Ny*Nz+2*Nx*Ny));

	cudaMemset(dyZ,0,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMemset(drZ,0,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMemset(dpZ,0,sizeof(T)*(nDV+2*NxZ*NyZ));
	cudaMemset(dqZ,0,sizeof(T)*(nDV+2*NxZ*NyZ));
}

void cudaFinalizeDeflation(T *&dpzc, T *&dpzf, T *&dpzs, T *&dpzw,
		T *&dec, T *&def, T *&des, T *&dew,
		T *&drZ, T *&dyZ, T *&dqZ, T *&dpZ,
		T *&drhs)
{
	cudaFree(dpzc);
	cudaFree(dpzf);
	cudaFree(dpzs);
	cudaFree(dpzw);
	cudaFree(dyZ);
	cudaFree(drZ);
	cudaFree(dpZ);
	cudaFree(dqZ);
	cudaFree(dec);
	cudaFree(def);
	cudaFree(des);
	cudaFree(dew);
	cudaFree(drhs);

}

// local DOT PRODUCT
// ---- y1 = Z' * y (works!) ----
template <class T, unsigned int blockSize>
__global__ void localDOTGPU(T *c,
		const T *a)
{
	int Nx  = d_dims.Nx;
	int Ny  = d_dims.Ny;
	int NxZ = d_paramsZ.NxZ;
	int NyZ = d_paramsZ.NyZ;
	int nRowsZ = d_paramsZ.nRowsZ;
	int NxZNyZ = NxZ*NyZ;
	int NxNy   = Nx*Ny;

	extern __shared__ T cache[];

	unsigned int iz =  blockIdx.x/(NxZNyZ);
	unsigned int iy =  (blockIdx.x%(NxZNyZ))/NxZ;
	unsigned int ix =   blockIdx.x%NxZ;

	unsigned int i  =  nRowsZ * (NxNy*iz + Nx*iy + ix);      // a global index of the first cell in a coarse cell  (parenthesis are necessary!)

	unsigned int tid  = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;			// local index within the coarse cell
    unsigned int ioff = NxNy + i + NxNy*threadIdx.z + Nx*threadIdx.y + threadIdx.x;

	cache[tid] = a[ioff];

	for (int j=1;j<blockDim.x/blockDim.z;j++)	cache[tid] += a[ioff + NxNy*blockDim.z*j];

	__syncthreads();

	if(blockSize >= 1024) {	if(tid < 512) { cache[tid] += cache[tid + 512]; } __syncthreads(); }
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

	if (tid == 0) c[blockIdx.x+NxZNyZ] = cache[0];
}

// ---- y (= Py) = y - AZ * y2 ----
template <class T>
__global__ void YMinusAzXYDeflation(T *y, // Nx*Ny*Nz threads must be launched!
		const T *x,
		const T *dpzc,
		const T *dpzf,
		const T *dpzs,
		const T *dpzw)
{
	int Nx  = d_dims.Nx;
	int Ny  = d_dims.Ny;
	int Nz  = d_dims.Nz;
	int NxZ = d_paramsZ.NxZ;
	int NyZ = d_paramsZ.NyZ;
	int nRowsZ = d_paramsZ.nRowsZ;
	int NxZNyZ = NxZ*NyZ;
	int NxNy   = Nx*Ny;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid<NxNy*Nz) {
		int kZ =  tid/(NxNy*nRowsZ);					// z-index of coarse system
		int jZ = (tid/(Nx*nRowsZ))%NyZ;				    // y-index
		int iZ = (tid%Nx)/nRowsZ;				        // x-index

		int idZ = NxZNyZ + kZ*NxZNyZ + jZ*NxZ + iZ;          	// linear index of course system

		y[tid+NxNy]  -=  + dpzc[tid]      * x[idZ]       		// center
						 + dpzw[tid]      * x[idZ-1]     		// west
						 + dpzw[tid+1]    * x[idZ+1]     		// east
						 + dpzs[tid]      * x[idZ-NxZ]   		// south
						 + dpzs[tid+Nx]   * x[idZ+NxZ]   		// north
						 + dpzf[tid]      * x[idZ-NxZNyZ]		// front
						 + dpzf[tid+NxNy] * x[idZ+NxZNyZ];		// back
	}
}

// ---- y (= P(^T)y) = y - Z * y2 ----
template <class T>
__global__ void YMinusZXYDeflation(T *y,
		const T *x)
{
	// Nx*Ny*Nz threads must be launched!
	int Nx  = d_dims.Nx;
	int Ny  = d_dims.Ny;
	int Nz  = d_dims.Nz;
	int NxZ = d_paramsZ.NxZ;
	int NyZ = d_paramsZ.NyZ;
	int nRowsZ = d_paramsZ.nRowsZ;
	int NxZNyZ = NxZ*NyZ;
	int NxNy   = Nx*Ny;
	int tid    = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid<NxNy*Nz) {
		int kZ =  tid/(NxNy*nRowsZ);					// z-index of coarse system
		int jZ = (tid/(Nx*nRowsZ))%NyZ;				    // y-index
		int iZ = (tid%Nx)/nRowsZ;				        // x-index

		int idZ = kZ*NxZNyZ + jZ*NxZ + iZ + NxZNyZ;   	// linear index of course system    + offset

		y[tid+NxNy] -= x[idZ];							// center
	}
}

// ---- y (= P(^T)*y + Q*b) = y + Z * y2 ----
template <class T>
__global__ void YPlusZXYDeflation(T *y,
		const T *x)
{
	// Nx*Ny*Nz threads must be launched!
	int Nx  = d_dims.Nx;
	int Ny  = d_dims.Ny;
	int Nz  = d_dims.Nz;
	int NxZ = d_paramsZ.NxZ;
	int NyZ = d_paramsZ.NyZ;
	int nRowsZ = d_paramsZ.nRowsZ;
	int NxZNyZ = NxZ*NyZ;
	int NxNy   = Nx*Ny;
	int tid    = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid<NxNy*Nz) {
		int kZ =  tid/(NxNy*nRowsZ);					// z-index of coarse system
		int jZ = (tid/(Nx*nRowsZ))%NyZ;				    // y-index
		int iZ = (tid%Nx)/nRowsZ;				        // x-index

		int idZ = kZ*NxZNyZ + jZ*NxZ + iZ + NxZNyZ;     // linear index of course system   + offset

		y[tid+NxNy] += x[idZ];						    // center
	}
}

#endif /* CUDAFUNCTIONSDEFLATION_H_ */
