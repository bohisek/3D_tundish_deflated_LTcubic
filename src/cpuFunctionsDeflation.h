#ifndef CPUFUNCTIONSDEFLATION_H_
#define CPUFUNCTIONSDEFLATION_H_

// initialize CPU fields
template <class T>
void cpuInitDeflation(T *&pzc, T *&pzf, T *&pzs, T *&pzw,
		T *&ec, T *&ef, T *&es, T *&ew,
		T *&pc, T *&pf, T *&ps, T *&pw,
		T *&lc, T *&lf, T *&ls, T *&lw,
		T *&hrZ, T *&hyZ, T *&hqZ, T *&hpZ, T *&hsZ,
		T *&L)
{

	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;
	int nDV = paramsZ.nDV;

	pzc = new T[Nx*Ny*Nz];      // A*Z ... constructed explicitly on CPU
	pzf = new T[Nx*Ny*Nz+Nx*Ny];
	pzs = new T[Nx*Ny*Nz+Nx];
	pzw = new T[Nx*Ny*Nz+1];

    ec = new T[nDV+2*NxZ*NyZ];      // E ... invertible Galerkin matrix
    ef = new T[nDV+NxZ*NyZ];
    es = new T[nDV+NxZ];
	ew = new T[nDV+1];

	pc = new T[Nx*Ny*Nz+2*Nx*Ny];      // A matrix of pressure on CPU
	pf = new T[Nx*Ny*Nz+Nx*Ny];
	ps = new T[Nx*Ny*Nz+Nx];
	pw = new T[Nx*Ny*Nz+1];

#ifdef DEFLDIR
	L = new T[nDV*(NxZ*NyZ+1)];
	memset(L,0,nDV*(NxZ*NyZ+1)*sizeof(T));
#else
	lc = new T[nDV+2*NxZ*NyZ];      // E ... invertible Galerkin matrix
	lf = new T[nDV+NxZ*NyZ];
	ls = new T[nDV+NxZ];
	lw = new T[nDV+1];

	memset(lc,0,(nDV+2*NxZ*NyZ)*sizeof(T));
	memset(lf,0,(nDV+NxZ*NyZ)*sizeof(T));
	memset(ls,0,(nDV+NxZ)*sizeof(T));
	memset(lw,0,(nDV+1)*sizeof(T));
#endif

	hrZ = new T[nDV+2*NxZ*NyZ];
	hyZ = new T[nDV+2*NxZ*NyZ];
	hqZ = new T[nDV+2*NxZ*NyZ];
	hpZ = new T[nDV+2*NxZ*NyZ];
	hsZ = new T[nDV+2*NxZ*NyZ];

	memset(pzc, 0, (Nx*Ny*Nz)*sizeof(T));
	memset(pzf, 0, (Nx*Ny*Nz+Nx*Ny)*sizeof(T));
	memset(pzs, 0, (Nx*Ny*Nz+Nx)*sizeof(T));
	memset(pzw, 0, (Nx*Ny*Nz+1)*sizeof(T));

	memset(pc, 0, (Nx*Ny*Nz+2*Nx*Ny)*sizeof(T));
	memset(pf, 0, (Nx*Ny*Nz+Nx*Ny)*sizeof(T));
	memset(ps, 0, (Nx*Ny*Nz+Nx)*sizeof(T));
	memset(pw, 0, (Nx*Ny*Nz+1)*sizeof(T));

	memset(ec, 0, (nDV+2*NxZ*NyZ)*sizeof(T));
	memset(ef, 0, (nDV+NxZ*NyZ)*sizeof(T));
	memset(es, 0, (nDV+NxZ)*sizeof(T));
	memset(ew, 0, (nDV+1)*sizeof(T));

	memset(hrZ, 0, (nDV+2*NxZ*NyZ)*sizeof(T));
	memset(hyZ, 0, (nDV+2*NxZ*NyZ)*sizeof(T));
	memset(hqZ, 0, (nDV+2*NxZ*NyZ)*sizeof(T));
	memset(hpZ, 0, (nDV+2*NxZ*NyZ)*sizeof(T));
	memset(hsZ, 0, (nDV+2*NxZ*NyZ)*sizeof(T));
}

template <class T>
void cpuFinalizeDeflation(T *&pzc, T *&pzf, T *&pzs, T *&pzw,
		T *&ec, T *&ef, T *&es, T *&ew,
		T *&pc, T *&pf, T *&ps, T *&pw,
		T *&lc, T *&lf, T *&ls, T *&lw,
		T *&hrZ, T *&hyZ, T *&hqZ, T *&hpZ, T *&hsZ,
		T *&L)
{
	delete[] pzc;
	delete[] pzf;
	delete[] pzs;
	delete[] pzw;

	delete[] pc;
	delete[] pf;
	delete[] ps;
	delete[] pw;

	delete[] ec;
	delete[] ef;
	delete[] es;
	delete[] ew;

#ifdef DEFLDIR
	delete[] L;
#else
	delete[] lc;
	delete[] lf;
	delete[] ls;
	delete[] lw;
#endif

	delete[] hrZ;
	delete[] hyZ;
	delete[] hqZ;
	delete[] hpZ;
	delete[] hsZ;


}


// initialize AZ (=A*Z); Z ... deflation subspace matrix
template <class T>
void initAZ(T *pzc, T *pzf, T *pzs, T *pzw,
		const T *pc, const T *pf, const T *ps, const T *pw)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;
	int nRowsZ = paramsZ.nRowsZ;

	// --- pzc, pzf, pzs, pzw ---
	for (int k=0; k<Nz; k++)
	{
		for (int j=0; j<Ny; j++)
		{
			for (int i=0; i<Nx; i++)
			{
				int id  = k*Nx*Ny + j*Nx + i;                   // linear system of original system
				//pzc                                              	_
				pzc[id] = pc[id+Nx*Ny];                 	    //   |
				if (i%nRowsZ!=0)     pzc[id] += pw[id];    		//   |
				if ((i+1)%nRowsZ!=0) pzc[id] += pw[id+1];  		//   |
				if (j%nRowsZ!=0)     pzc[id] += ps[id];    		//   |
				if ((j+1)%nRowsZ!=0) pzc[id] += ps[id+Nx];		//   |
				if (k%nRowsZ!=0)     pzc[id] += pf[id];    		//   |
				if ((k+1)%nRowsZ!=0) pzc[id] += pf[id+Nx*Ny];  	//   |
				//pzw                                              		  >  compact and works fine!
				if (i%nRowsZ==0) 	 pzw[id]  = pw[id];   		//   |
				//pzs                                                |
				if (j%nRowsZ==0)     pzs[id]  = ps[id];    		//   |
				//pzf                                              	 |
				if (k%nRowsZ==0)     pzf[id]  = pf[id];    		//  _|
			}
		}
	}
}


// initialize E (Rspace = nDVxnDV); E ... invertible Galerkin matrix
template <class T>
void initE(T *ec, T *ef, T *es, T *ew,
		const T *pc, const T *pf, const T *ps, const T *pw)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int Nz = dims.Nz;
	int nRowsZ = paramsZ.nRowsZ;
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;

	for (int k=0; k<Nz; k++)
	{
		for (int j=0; j<Ny; j++)                 // works just fine!
		{
			for (int i=0; i<Nx; i++)
			{
				int iZ = i/nRowsZ;						// x-index of coarse system
				int jZ = j/nRowsZ;						// y-index
				int kZ = k/nRowsZ;						// z-index
				int idZ = kZ*NxZ*NyZ + jZ*NxZ + iZ;		// linear index of course system
				int id  = k*Nx*Ny + j*Nx + i;			// linear system of original system

				// EC
				ec[idZ+NxZ*NyZ] += pc[id+Nx*Ny];
				if (k%nRowsZ!=0)	ec[idZ+NxZ*NyZ] += 2*pf[id];
				// adding pw as well as ps and pf could be avoided if pc did NOT include fluxes!
				if (j%nRowsZ!=0)	ec[idZ+NxZ*NyZ] += 2*ps[id];
				// 2*pw due to the fact that for one cell it is WEST flux, for the neighbor it is EAST flux.
				if (i%nRowsZ!=0)	ec[idZ+NxZ*NyZ] += 2*pw[id];
				//EF
				if (k%nRowsZ==0)	ef[idZ] += pf[id];
			    //ES
				if (j%nRowsZ==0)	es[idZ] += ps[id];
				//EW
				if (i%nRowsZ==0)	ew[idZ] += pw[id];
			}
		}
	}
}


// "spChol" MUST be correct, tested against OCTAVE!

#ifdef DEFLDIR
template <class T>
void Chol(T *L, const T *ec, const T *ef, const T *es, const T *ew)
{
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;
	int nDV = paramsZ.nDV;
	int NxZNyZ = NxZ*NyZ;
	T sumL2jk, sumLikLjk;

	for (int i=0; i<nDV; i++)
		{
			int ixm = min(NxZNyZ,i);
			sumL2jk = 0.;							// reset sum of squares of Ljk for forthcoming Lj,j calculation

			for (int j=0; j<ixm; j++)
			{
				int jr = int(max(i-NxZNyZ, 0)) + j;// row index of L (each diagonal starting at first row!)
				int jy = ixm - j;					// column index of L
				int idL = jr + nDV * jy;            // linear index of L
				sumLikLjk = 0.;						// reset sum of squares of Lik*Ljk

				 for (int k=j-1; k>=0; k--)
				 {
					 int jrk = jr - k - 1;   // for Lik as well as Ljk
					 int jyi = jy + k + 1;     // Lik
					 int jyj =    + k + 1;     // Ljk
					 sumLikLjk += L[jrk+nDV*jyi] * L[jrk+nDV*jyj];
				 }
				 switch(jy)
				 {
				 case 1:
					 L[idL] = 1/L[jr]  * (ew[jr+1]       - sumLikLjk);
					 break;
				 case 16:   // NxZ 32
					 L[idL] = 1/L[jr]  * (es[jr+NxZ]     - sumLikLjk);
					 break;
				 case 64:  // NxZ*NyZ 256
					 L[idL] = 1/L[jr]  * (ef[jr+NxZNyZ]  - sumLikLjk);
					 break;
				 default:
					 L[idL] = 1/L[jr]  * (               - sumLikLjk);

				 }
				 sumL2jk = sumL2jk + L[idL]*L[idL];
			}
			L[i] = sqrt(ec[i+NxZNyZ] - sumL2jk);
		}
}

// "solve" MUST be correct, tested against OCTAVE!
// DC: direct solver with Cholesky decomposition
template <class T>
void solveDC(T *hyZ, const T *hrZ, const T *L)
{
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;
	int nDV = paramsZ.nDV;
	int NxZNyZ = NxZ*NyZ;

	// forward sweep
	for (int i=0; i<nDV; i++)
	{
		hyZ[i+NxZNyZ] = hrZ[i+NxZNyZ];
		int ixm = min(NxZNyZ,i);
		int iym = max(i-NxZNyZ, 0);

		for (int j=0; j<ixm; j++)
		{
			hyZ[i+NxZNyZ] -= L[iym+j+nDV*(ixm-j)] * hyZ[iym+j+NxZNyZ];
		}
		hyZ[i+NxZNyZ] /= L[i];
	}

	// back sweep
	for (int i=nDV-1; i>=0; i--)
	{
		for (int j=0; j<NxZNyZ; j++)
		{
			hyZ[i+NxZNyZ] -= L[i + nDV * (j+1)] * hyZ[i+j+1+NxZNyZ];
		}
		hyZ[i+NxZNyZ] /= L[i];
	}
}
#endif

// ---------------------ICCG--------------------------------
#ifndef DEFLDIR
// diagonal incomplete Cholesky IC(0) A = L*D*L^T
// modified diagonal incomplete Cholesky MIC(0)
// lc ... diagonal D (or P in paper: HIGH-PERFORMANCE PCG SOLVERS FOR FEM STRUCTURAL ANALYSIS)
// lw, ls, lf ... subdiagonals of L.
// Note: diagonal of L not stored, because all elements = 1
// (see OCTAVE: icholTest.m or icholTest3D.m)
// checked with OCTAVE eSolve3D_16x4x4.m
template <class T>
void IChol(T *lc, T *lf, T *ls, T *lw,
		const T *ec, const T *ef, const T *es, const T *ew)

{
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;
	int nDV = paramsZ.nDV;
	int NxZNyZ = NxZ*NyZ;
	int sizeZ = nDV + 2*NxZNyZ;
	T temp;
	// lc
	memcpy(lc, ec, sizeof(T)*sizeZ);	// lc := ec
	int ids = NxZNyZ;
	for (int id=0;id<nDV-1;id++,ids++)
	{
		temp            = ew[id+1]/lc[ids];
		lc[ids+1]      -= temp*ew[id+1];
#ifdef MIC0 // modified incomplete Cholesky with zero fill
		lc[ids+1]      -= temp*es[id+NxZ];
		lc[ids+NxZ]    -= temp*es[id+NxZ];
		lc[ids+1]      -= temp*ef[id+NxZNyZ];
		lc[ids+NxZNyZ] -= temp*ef[id+NxZNyZ];
#endif
		temp            = es[id+NxZ]/lc[ids];
		lc[ids+NxZ]    -= temp*es[id+NxZ];
#ifdef MIC0
		lc[ids+NxZ]    -= temp*ef[id+NxZNyZ];
		lc[ids+NxZNyZ] -= temp*ef[id+NxZNyZ];
#endif
		temp            = ef[id+NxZNyZ]/lc[ids];
		lc[ids+NxZNyZ] -= temp*ef[id+NxZNyZ];

		// lw, ls, lf
		lw[id+1]        = ew[id+1]/lc[ids];
		ls[id+NxZ]      = es[id+NxZ]/lc[ids];
		lf[id+NxZNyZ]   = ef[id+NxZNyZ]/lc[ids];
	}
}


template <class T>
void solveIC(T *hsZ, const T *hrZ,
		     const T *ec, const T *ef, const T *es, const T *ew,
		     const T *lc, const T *lf, const T *ls, const T *lw)	// for solveICCG
{
	int NxZ = paramsZ.NxZ;
    int NyZ = paramsZ.NyZ;
    int NxZNyZ = NxZ*NyZ;
    int nDV = paramsZ.nDV;
    int ids = NxZNyZ;
    // note: id = ids-NxZNyZ
    // A = L*D*L^T, L*y=b ... forward sweep
    for (int id=0; id<nDV; id++, ids++)	// L*y1 = r => y1
    	hsZ[ids] =  hrZ[ids] - lw[id]*hsZ[ids-1] - ls[id]*hsZ[ids-NxZ] - lf[id]*hsZ[id] ;
    // D*L^T*x = y ... backward sweep
    // D*L^T properties: main diagonal = D, superdiagonals identical to those of original A
    //for (int i=0; i<nDV; i++) cout << hsZ[i+NxZNyZ] << endl;

    //cout << endl << "ids: " << ids << endl;
    --ids;

    for (int id=nDV-1; id>=0; id--, ids--)	// D*L^T*y2 = y1 => y2
        hsZ[ids] = ( hsZ[ids] - ew[id+1]*hsZ[ids+1] - es[id+NxZ]*hsZ[ids+NxZ] - ef[ids]*hsZ[ids+NxZNyZ] ) / lc[ids];

    //for (int i=0; i<nDV; i++) cout << hsZ[i+NxZNyZ] << endl;

    //cout << endl << "---DONE---" << endl << endl;
}


// ICCG: incomplete Cholesky CG
template <class T>
void solveICCG(T *hsZ, T *hrZ, T *hyZ, T *hpZ, T *hqZ,
	     const T *ec, const T *ef, const T *es, const T *ew,
	     const T *lc, const T *lf, const T *ls, const T *lw)
{
	int NxZ = paramsZ.NxZ;
	int NyZ = paramsZ.NyZ;
	int NxZNyZ = NxZ*NyZ;
	int nDV = paramsZ.nDV;
	int sizeZ = nDV + 2*NxZNyZ;
	T maxresZ = paramsZ.maxresZ;

	memcpy(hyZ, hrZ, sizeof(T)*sizeZ);
	// r := b - Ax
	int ids = NxZNyZ;  // tid shifted
	for (int id=0; id<nDV; id++, ids++)
	{
		hrZ[ids] -= ec[ids]    * hyZ[ids]         // center
		          + es[id+NxZ] * hyZ[ids+NxZ]     // north               N
			      + ew[id+1]   * hyZ[ids+1]       // east              W C E
			      + es[id]     * hyZ[ids-NxZ]     // south               S
			      + ew[id]     * hyZ[ids-1]       // west
			      + ef[ids]    * hyZ[ids+NxZNyZ]  // back                B
			      + ef[id]     * hyZ[id];         // front               F
	}
	// rhoNew = r^T*r
	T rhOldZ, btZ, sgZ, apZ;
	T rhNewZ = 0.; // reset
	solveIC(hsZ,hrZ,ec,ef,es,ew,lc,lf,ls,lw);   // z = M^(-1)r
	ids = NxZNyZ;
	for (int id=0; id<nDV; id++, ids++)	rhNewZ += hsZ[ids]*hrZ[ids];
	// stopping criterion
    T stopZ = rhNewZ * maxresZ * maxresZ;
    //cout << "stopZ:" << stopZ << ", residualZ: " << rhNewZ;
    int iterZ = 0;
    // loop
    while (rhNewZ > stopZ)
    {
    	iterZ++;
    	//cout << "iterationZ:" << iterZ << ", residualZ: " << rhNewZ << endl;
    	if (iterZ==1)	memcpy(hpZ, hsZ, sizeof(T)*sizeZ); // p :=z
    	else {
    		btZ = rhNewZ/rhOldZ;
    		ids = NxZNyZ;
    		for (int id=0; id<nDV; id++, ids++)	hpZ[ids] = hsZ[ids] + btZ*hpZ[ids];  // p := r + beta*p
    	}
    	// q := Ap
    	ids = NxZNyZ;
    	for (int id=0; id<nDV; id++, ids++)
    	{
    		hqZ[ids] = ec[ids]    * hpZ[ids]           // center
    				 + es[id+NxZ] * hpZ[ids+NxZ]       // north               N
    			     + ew[id+1]   * hpZ[ids+1]         // east              W C E
    				 + es[id]     * hpZ[ids-NxZ]       // south               S
    				 + ew[id]     * hpZ[ids-1]         // west
    			     + ef[ids]    * hpZ[ids+NxZNyZ]    // back                B
    				 + ef[id]     * hpZ[id];           // front               F
    	}
    	// sigma := p^T*q
    	sgZ = 0.; // reset
    	ids = NxZNyZ;
    	for (int id=0; id<nDV; id++, ids++)	sgZ += hpZ[ids]*hqZ[ids];
    	// alpha := rhoNew/sigma
    	apZ = rhNewZ/sgZ;
    	ids = NxZNyZ;
    	for (int id=0; id<nDV; id++, ids++) {
    		hrZ[ids] -= apZ*hqZ[ids]; // r := r - alpha*q
    		hyZ[ids] += apZ*hpZ[ids]; // x := x + alpha*p
    	}
    	solveIC(hsZ,hrZ,ec,ef,es,ew,lc,lf,ls,lw); // z = M^(-1)r
    	rhOldZ = rhNewZ;
    	// rhoNew = r^T*r
    	rhNewZ = 0.; // reset
    	ids = NxZNyZ;
    	for (int id=0; id<nDV; id++, ids++)	rhNewZ += hsZ[ids]*hrZ[ids];
    }
    //cout << ", iterationZ:" << iterZ << endl;
}

#endif
#endif /* CPUFUNCTIONSDEFLATION_H_ */
