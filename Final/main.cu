#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <sys/time.h> 

#define THREAD_GRAIN 8
#define BLOCK_SIZE 256

#define EPSILON 1e-6
#define COMP_EPSILON 1e-8
//#define ENABLE_FLATTEN_MATRICES
#define ENABLE_FLIPPED_MATRICES

// Sequential implementation of Step 2
void StepTwoGPADFlatSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m){
	
	for (int i = 0; i < N; i++){
		for (int j = 0; j < n_u; j++){ 
			float sum = 0.0; 
			for (int k = j; k < 4*n_u*N; k += n_u){
				sum += M_G[i*m + k]*w_v[k];  
			}
			for (int k = 4*n_u*N; k < m; k++){
				sum += M_G[i*m + k]*w_v[k]; 
			}
			zhat_v[i*n_u + j] = sum - g_P[i*n_u + j];
		}
	}

}

// Sequential implementation of Step 4
void StepFourGPADFlatSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	for (int i = 0; i < m; i++){
		float sum = 0.0f; 
		for (int j = 0; j < N; j++){
			if (i < 4*n_u*N){
				sum += G_L[i*N + j]*zhat_v[j*n_u + (i%n_u)]; 
			}
			else{
				for(int k = 0; k < n_u; k++){
					sum += G_L[i*N + j]*zhat_v[j*n_u + k]; 
				}
			}
		}
		y_vp1[i] = sum + w_v[i] + p_D[i]; 
	}
	
	for(int i = 0; i < m; i++){
		if(y_vp1[i] < 0) y_vp1[i] = 0;
	}
}

// Sequential implementation of Step 2 over unflattened matrices 
void StepTwoGPADSequential(const float* M_G, float* w_v, const float* g_P, float* zhat_v, const int N, const int n_u, const int m){
	
	int numrows_M_G = n_u*N;
	int numcols_M_G = m; 
	for (int i = 0; i < numrows_M_G; i++){
		float sum = 0.0f; 
		for(int j = 0; j < numcols_M_G; j++){
			sum += M_G[i*numcols_M_G + j]*w_v[j]; 
		}
		zhat_v[i] = sum - g_P[i]; 
	}

}

void StepThreeGPADSequential(const float theta, float* zhat_v, float* z_v, const int length){
	for (int i = 0; i < length; i++){
		z_v[i] += theta* (zhat_v[i] - z_v[i]); // single fused multiply-add instruction
	}
}

// Sequential implementation of Step 4 over unflattened matrices 
void StepFourGPADSequential(const float* G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	int numrows_G_L = m;
	int numcols_G_L = n_u*N;
	for (int i = 0; i < numrows_G_L; i++){
		float sum = 0.0f; 
		for (int j = 0; j < numcols_G_L; j++){
			sum += G_L[i*numcols_G_L + j]*zhat_v[j]; 
		}
		sum += w_v[i] + p_D[i]; 
		y_vp1[i] = (sum + abs(sum))/2; 
	}
}

__global__ void StepOneGPADKernel(float *y_vec_in, float *y_vec_minus_1_in, float *w_vec_out, float beta_v, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        w_vec_out[i] = y_vec_in[i] + beta_v * (y_vec_in[i] - y_vec_minus_1_in[i]);
    }
}

__global__ void StepTwoGPADKernel(const float* __restrict__ M_G, float* w_v, float* g_P, float* zhat, int N, int n_u, int m) 
{
	extern __shared__ float w_vs[]; 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elements = N * n_u;
	int index = 0; 
	while(threadIdx.x + index < m){
		w_vs[threadIdx.x + index] = w_v[threadIdx.x + index]; // coalesced memory accesses 
		index += blockDim.x; 
	}
	__syncthreads(); 
	
    if (i < total_elements) {
        float sum = 0.0;
        for (int j = 0; j < m; j++) {
            sum += M_G[j + i * m] * w_vs[j];
        }
        zhat[i] = sum - g_P[i];
    }
}

__global__ void StepThreeGPADKernel(float theta, float* zhat_v, float* z_v, int length) {
    
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        z_v[i] = z_v[i] + theta * (zhat_v[i] - z_v[i]);
    }
}

__global__ 
void StepThreeGPADKernelVectorized(float theta, float* zhat_v, float* z_v, int length) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int index = tid * 4;

    if (index + 4 < length) {
        float4 zhat_v4 = reinterpret_cast<const float4*>(&zhat_v[index])[0];
		float4 z_v4 = reinterpret_cast<float4*>(&z_v[index])[0];
        z_v[index + 0] += theta * (zhat_v4.x - z_v4.x);
		z_v[index + 1] += theta * (zhat_v4.y - z_v4.y);
		z_v[index + 2] += theta * (zhat_v4.z - z_v4.z);
		z_v[index + 3] += theta * (zhat_v4.w - z_v4.w);
    }
}

__global__ void StepFourGPADFlatParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* __restrict__ p_D, float* zhat_v, const int N, const int n_u, const int m){
	
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[]; 
	int tx = threadIdx.x; 
	int bx = blockIdx.x; 
	int index = tx + bx*blockDim.x; 
	
	// collaborate in loading the shared memory with m threads
	int i = 0; 
	while(tx + i < n_u*N){
		zhat_vs[tx + i] = zhat_v[tx + i]; // coalesced memory accesses 
		i += blockDim.x; 
	}
	__syncthreads(); 
	
	// handle out of bounds 
	if(index < m){
		float sum = 0.0f; 
		float yraw = 0.0f; 
		for (int j = 0; j < N; j++){
			if (index < 4*n_u*N){
				sum += G_L[index*N + j]*zhat_vs[j*n_u + (index % n_u)]; 
			}
			else{
				for(int k = 0; k < n_u; k++){
					sum += G_L[index*N + j]*zhat_vs[j*n_u + k]; 
				}
			}
		}
		yraw = sum + w_v[index] + p_D[index]; // sum 
		
		y_vp1[index] = (yraw < COMP_EPSILON) ? 0 : yraw; // projection onto nonnegative orthant
	}

}

__global__ void StepFourGPADFlippedParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m, int max_threads){
	
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[];  
	int index = threadIdx.x + blockIdx.x*blockDim.x; 
	int numcols_G_L = n_u*N;
	
	// collaborate in loading the shared memory with m threads
	for(int i = threadIdx.x; i < numcols_G_L; i+= blockDim.x){
		zhat_vs[i] = zhat_v[i]; // coalesced memory accesses
	}
	__syncthreads(); 
		
	// handle out of bounds 
	for (int i = index; i < m; i += gridDim.x*blockDim.x){
		float sum = 0.0f; 
		for (int j = 0; j < numcols_G_L; j++)
		{
			sum += G_L[j*m + i]*zhat_vs[j]; // flipping the matrices results in coalesced memory accesses
		}
		sum += w_v[i] + p_D[i]; // sum 
		y_vp1[i] = (sum + abs(sum))/2; // max without control divergence
		//y_vp1[i] = (sum < COMP_EPSILON) ? 0 : sum;
	}
}

__global__ void DeviceArrayCopy(float* dest, float* src, int size){
	
	int index = threadIdx.x + blockIdx.x * blockDim.x; 
	if (index < size){ dest[index] = src[index]; }
}


__global__ void StepFourGPADParRows(const float* __restrict__ G_L, float* y_vp1, float* w_v, const float* p_D, float* zhat_v, const int N, const int n_u, const int m, int max_threads){
		
	// launch m threads to compute a unique element of the output vector 
	
	extern __shared__ float zhat_vs[];
	int tx = threadIdx.x; 
	int bx = blockIdx.x; 
	int index = tx + bx*blockDim.x; 
	int numcols_G_L = n_u*N; 
	
	// collaborate in loading the shared memory with m threads
	for(int i = tx; i < numcols_G_L; i+= blockDim.x){
		zhat_vs[i] = zhat_v[i]; // coalesced memory accesses
	}
	__syncthreads(); 
		
	if (index < max_threads){	
		// handle out of bounds 
		for (int i = index; i < m; i += gridDim.x*blockDim.x){
			float sum = 0.0f; 
			for (int j = 0; j < numcols_G_L; j++)
			{
				sum += G_L[i*numcols_G_L + j]*zhat_vs[j]; 
			}
			sum += w_v[i] + p_D[i]; // sum 
			y_vp1[i] = 0.5*(sum + abs(sum)); // max without control divergence
		}
	}

}

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ 
void mv_warpReduceNoRollOverGranular(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ d, float* c, int m, int n) {
    // designed for datasets with more then 1024 columns
    extern __shared__ float warpSum[]; // Dynamic shared memory for warp sums

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rowLength = ((n + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int row = (blockIdx.x / ((n + blockDim.x - 1) / blockDim.x)) * THREAD_GRAIN;
    int col = tid % (rowLength);
    int lane = tid % 32;   // Lane index within the warp
    int warpIdx = (col / 32); // Warp index within the block
    int warpIdxLocal = (warpIdx % 32) % (blockDim.x / 32); // warp index within a block 0 - 31

    float bLocal = b[col];
    for (int phase = 0; phase < THREAD_GRAIN; phase++){
        if (row + phase > m) return; // Bounds check for rows

        float localVal = 0.0f;

        if (tid < 32) warpSum[tid] = 0.0f;

        if (col < n) {
            localVal += a[(row + phase) * n + col] * bLocal;
        }

        // Warp-level reduction
        localVal = warpReduceSum(localVal);

        // Store results in shared memory
        if (lane == 0) warpSum[warpIdxLocal] = localVal;

        __syncthreads();

        // Reduce warp results to a single value
        if (warpIdxLocal == 0 ) {
            float sum = 0.0f;
            localVal = (lane < 32) * warpSum[lane];
            sum = warpReduceSum(localVal);

            if (lane == 0) atomicAdd(&c[row + phase], sum);
			if (col == 0) atomicAdd(&c[row + phase], d[row + phase]);
        }

    }
}

__global__
void mv_warpReduce(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ d, float* c, int m, int n) {
    // designed for datasets with more then 1024 columns
    extern __shared__ float warpSum[]; // Dynamic shared memory for warp sums

    // The compliler will optimize these calculations and their register usage
    // Each thread can have up to 256 registers in the and this implemetation doesn't go above 10 registers per thread
    int warpsPerRow = (n + 31) / 32; // rounds up to the nearest multiple of 32
    int warpsPerBlock = (blockDim.x + 31) / 32; // rounds up to the nearest multiple of 32 

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int col = (tid % (warpsPerRow * 32));// Column index
    int row = tid / (warpsPerRow * 32); // Row index (global row index)
    int lane = col % 32;   // Lane index within the warp

    int firstRow = (blockIdx.x * blockDim.x ) / (warpsPerRow * 32); // First row in the block
    int firstWarp = ((blockIdx.x * blockDim.x + 31) - row * warpsPerRow * 32) / 32; // First warp in the block

    int warpIdx = (col / 32) % warpsPerRow; // Warp index within the block
    int warpIdxLocal = (warpIdx % 32); // warp index within a block 0 - 31
    int warpIdy = row;  // Starts counting from zero for each block (local row index)
    int warpIdyLocal = warpIdy - firstRow ;


    //fills the shared memory with zeros we should never have more than 32 warps per block (1024 threads)
    // int sharedMemSize = 64;
    if (threadIdx.x < 64) {
        warpSum[threadIdx.x] = 0.0f;
    }

    if (row >= m) return; // Bounds check for rows

    
    float localVal = 0.0f;

    localVal += a[row * n + col] * b[col];
    

    // Warp-level reduction
    localVal = warpReduceSum(localVal);

    // Store results in shared memory
    if (lane == 0) {
        // second row will always be stored starting at index 32
        warpSum[warpIdxLocal + warpIdyLocal * 32] = localVal;
    }

    __syncthreads();

    // Reduce warp results to a single value
    if (warpIdx == firstWarp || warpIdx == 0) {
        float sum = 0.0f;
        
        // warpIdx has a row associated with it

        localVal = (lane < warpsPerRow) * warpSum[lane + warpIdyLocal * 32];
        sum = warpReduceSum(localVal);
        // printf("warpIdx %d, warpIdxLocal %d, warpIdy %d, sum %f  lane %d blockidx.x %d, row %d\n", warpIdx, warpIdxLocal, warpIdy, sum, lane, blockIdx.x, row);
        

        if (lane == 0) {
            // printf("warpIdx %d, warpIdxLocal %d, warpIdy %d, sum %f  lane %d blockidx.x %d, row %d\n", warpIdx, warpIdxLocal, warpIdy, sum, lane, blockIdx.x, row);
            atomicAdd(&c[row], sum);
        }
		if (col == 0) atomicAdd(&c[row], d[row]);
    }
}

// Function to print a vector
void printVector(const float* vec, int size, const char* name) {
    printf("%s: [", name);
    for (int i = 0; i < size; ++i) {
        printf("%f", vec[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

// Function to print a matrix
void printMatrix(const float* mat, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vectors_side_by_side(const float* vector1, const float* vector2, int length) {
    printf("Vector1\t\tVector2\n");
    printf("-------\t\t-------\n");
    for (int i = 0; i < length; ++i) {
        printf("%.8f\t\t%.8f\n", vector1[i], vector2[i]);
    }
}

void readData(const char *filename, int *n_u, int *N, int *m, int *num_iterations, float *L, float **M_G, float **g_P, float **G_L, float **p_D, float **theta, float **beta){
	
	// open files and read 
	FILE* file = fopen(filename, "r"); 
	
	if (fscanf(file, "%d %d %d %d %f", n_u, N, m, num_iterations, L) != 5) {
        perror("Error reading data");
        fclose(file);
    }
	
	#ifdef ENABLE_FLATTEN_MATRICES
        *M_G = (float*)calloc((*N) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N)*(*m); cnt++) fscanf(file, "%f", &(*M_G)[cnt]);
    #else
        *M_G = (float*)calloc((*N) * (*n_u) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N)*(*n_u)*(*m); cnt++) fscanf(file, "%f", &(*M_G)[cnt]);
    #endif
	
	*g_P = (float*)calloc((*N) * (*n_u), sizeof(float)); 
	for(int cnt = 0; cnt < (*N) * (*n_u); cnt++) fscanf(file, "%f", &(*g_P)[cnt]); 
	
	#ifdef ENABLE_FLATTEN_MATRICES
        *G_L = (float*)calloc((*N) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N) * (*m) cnt++) fscanf(file, "%f", &(*G_L)[cnt]);
    #else
        *G_L = (float*)calloc((*N) * (*n_u) * (*m), sizeof(float));
		for(int cnt = 0; cnt < (*N) * (*n_u) * (*m); cnt++) fscanf(file, "%f", &(*G_L)[cnt]);
    #endif
	
	*p_D = (float*)calloc(*m, sizeof(float)); 
	for(int cnt = 0; cnt < *m; cnt++) fscanf(file, "%f", &(*p_D)[cnt]); 
	
	*theta = (float*)calloc(*num_iterations, sizeof(float)); 
	for (int cnt = 0; cnt < *num_iterations; cnt++) fscanf(file, "%f", &(*theta)[cnt]); 
	*beta = (float*)calloc(*num_iterations, sizeof(float)); 
	for (int cnt = 0; cnt < *num_iterations; cnt++) fscanf(file, "%f", &(*beta)[cnt]);
	fclose(file);
	
}

void initializeVariables(float **y_v, float **y_vp1, float **w_v, float **zhat_v, float **z_v, int n_u, int N, int m){
	
	*y_v = (float*)calloc(m, sizeof(float)); 
	*y_vp1 = (float*)calloc(m, sizeof(float)); 
	*w_v = (float*)calloc(m, sizeof(float)); 
	*zhat_v = (float*)calloc(n_u*N, sizeof(float)); 
	*z_v = (float*)calloc(n_u*N, sizeof(float)); 
	
}


void flipRows(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows / 2; ++i) {
		for (int j = 0; j < cols; ++j) {
			float temp = matrix[i * cols + j];
			matrix[i * cols + j] = matrix[(rows - 1 - i) * cols + j];
			matrix[(rows - 1 - i) * cols + j] = temp;
		}
	}
}

int main(){
	
	// System variables 
	int n_u;
	int N;
	int m;
	int num_iterations; 
	float L; 
	
	// Constants computed off-line 
	float *M_G; 
	float *g_P; 
	float *G_L;
	float *p_D;
	float *theta; 
	float *beta; 
	
	// Variables computed on-line  
	float *y_vp1;
	float *y_v; 
	float *w_v;
	float *zhat_v;
	float *z_v; 
	
	// Read in data from text file and initialize variables 
	char filename[256];	
	snprintf(filename, sizeof(filename), "inputs_gpad/input_big.txt");
	readData(filename, &n_u, &N, &m, &num_iterations, &L, &M_G, &g_P, &G_L, &p_D, &theta, &beta); 
	initializeVariables(&y_v, &y_vp1, &w_v, &zhat_v, &z_v, n_u, N, m); 
	
	//printMatrix(M_G, n_u*N, m, "M_G"); 
	//printMatrix(G_L, m, n_u*N, "G_L"); 
	//printMatrix(g_P, n_u*N, 1, "g_P"); 
	//printMatrix(p_D, m, 1, "p_D"); 
	//printMatrix(theta, num_iterations, 1, "theta"); 
	//printMatrix(beta, num_iterations, 1, "beta");	
	
	// Write algorithm here!
	int v = 0; 
	long gpu_exectime = 0;
	
	// STEP 1
	float *dy_vp1; 
	float *dy_v; 
	float *dM_G;
	float *dg_P;
	float *dw_v; 
	float *dz_v; 
	float *dzhat_v;
	float* dp_D;
	float* dG_L;
	cudaMalloc((void **)&dy_vp1, m * sizeof(float)); 
	cudaMalloc((void **)&dy_v, m * sizeof(float)); 
	cudaMalloc((void **)&dM_G, N * n_u * m * sizeof(float));
	cudaMalloc((void **)&dg_P, N * n_u * sizeof(float));
	cudaMalloc((void **)&dw_v, m * sizeof(float)); 
	cudaMalloc((void **)&dz_v, n_u * N * sizeof(float)); 
	cudaMalloc((void **)&dzhat_v, n_u * N * sizeof(float));
	cudaMalloc((void**)&dp_D, m*sizeof(float));

	#ifdef ENABLE_FLATTEN_MATRICES
		cudaMalloc((void**)&dG_L, N*m*sizeof(float)); 
		cudaMemcpy(dG_L, G_L, N*m*sizeof(float), cudaMemcpyHostToDevice);
	#else 
		cudaMalloc((void**)&dG_L, N*n_u*m*sizeof(float)); 
		cudaMemcpy(dG_L, G_L, N*n_u*m*sizeof(float), cudaMemcpyHostToDevice);
	#endif

	cudaMemcpy(dy_vp1, y_vp1, m * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dy_v, y_v, m * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dM_G, M_G, N * n_u * m * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dg_P, g_P, N * n_u * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dz_v, z_v, n_u * N * sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(dzhat_v, zhat_v, n_u * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dp_D, p_D, m*sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 gridDimStep1(ceil((float)m / (float)256.0f), 1 , 1);
    dim3 blockDimStep1(256, 1, 1);

    int warpsPerRow = (N + 31) / 32;
    int warpsPerBlock = (BLOCK_SIZE + 31) / 32;
	dim3 gridDimStep2_warp((m * n_u * warpsPerRow + warpsPerBlock - 1) / warpsPerBlock);

	dim3 gridDimStep2((m * n_u / THREAD_GRAIN) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE));
	dim3 blockDimStep2(BLOCK_SIZE);

	dim3 gridDimStep3(ceil((float)n_u*N / (256.0f)), 1, 1); 
	dim3 gridDimStep3_vectorized(ceil((float)n_u*N / (256.0f * 4)), 1, 1);
	dim3 blockDimStep3(256, 1, 1);  

	dim3 gridDimStep4((int)ceil((float)m/256.0), 1, 1);
	dim3 blockDimStep4(256, 1, 1); 

	dim3 gridDimCopy((int)ceil((float)m/256.0), 1, 1);
	dim3 blockDimCopy(256, 1, 1); 
	
	for (v = 0; v < 1; v++){

		// STEP 1
		StepOneGPADKernel<<<gridDimStep1, blockDimStep1>>>(dy_vp1, dy_v, dw_v, beta[v], m); 
		cudaDeviceSynchronize();
		// STEP 2 & COPY STEP
		mv_warpReduce<<<gridDimStep2_warp, blockDimStep2, 64*sizeof(float)>>>(dM_G, dw_v, dg_P, dzhat_v, (n_u * m), N); 
		// mv_warpReduceNoRollOverGranular<<<gridDimStep2, blockDimStep2, 64*sizeof(float)>>>(dM_G, dw_v, dg_P, dzhat_v, n_u, N);
		DeviceArrayCopy<<<gridDimCopy, blockDimCopy>>>(dy_v, dy_vp1, m); 
		cudaDeviceSynchronize(); 
		// STEP 3 & STEP 4 
		StepThreeGPADKernelVectorized<<<gridDimStep3_vectorized, blockDimStep3>>>(theta[v], dzhat_v, dz_v, n_u*N); 
		StepThreeGPADKernel<<<gridDimStep3, blockDimStep3>>>(theta[v], dzhat_v, dz_v, n_u*N); 
		StepThreeGPADSequential(theta[v], zhat_v, z_v, n_u*N);

		StepFourGPADFlippedParRows<<<gridDimStep4, blockDimStep4, n_u*N*sizeof(float)>>>(dG_L, dy_vp1, dw_v, dp_D, dzhat_v, N, n_u, m, 3660);
		cudaDeviceSynchronize(); 

	}
	cudaMemcpy(y_vp1, dy_vp1, m * sizeof(float), cudaMemcpyDeviceToHost); 
	cudaMemcpy(zhat_v, dzhat_v, N*n_u*sizeof(float), cudaMemcpyDeviceToHost); 
	cudaMemcpy(w_v, dw_v, m * sizeof(float), cudaMemcpyDeviceToHost);
	//printMatrix(w_v, m, 1, "w_v"); 
	printMatrix(zhat_v, n_u*N, 1, "z_v"); 
	//printMatrix(y_vp1, m, 1, "Computed y_vp1");
	print_vectors_side_by_side(p_D, y_vp1, m); 
	cudaFree(dy_vp1); 
	cudaFree(dy_v); 
	cudaFree(dw_v); 
	cudaFree(dz_v); 
	cudaFree(dzhat_v);
	cudaFree(dp_D);
	
	printf("n_u = %d, N = %d, m = %d\n", n_u, N, m); 
	// printf("Total GPU Execution Time over %d trial(s) = %lu usec\n", 100, gpu_exectime); 
	// printf("Avg. GPU Execution Time over %d trial(s) = %lu usec\n", 100, gpu_exectime/100);
	
	// Free memory from the heap
	free(M_G);  
	free(g_P);
	free(zhat_v); 
	free(w_v); 
	free(p_D); 
	free(G_L); 
	free(y_vp1);
	free(theta);
	free(beta);
	free(z_v); 	
	
	return 0; 
}