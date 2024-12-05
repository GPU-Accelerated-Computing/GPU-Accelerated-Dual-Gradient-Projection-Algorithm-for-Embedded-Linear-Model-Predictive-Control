#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <cstdio>

#define THREAD_GRAIN 5
#define BLOCK_SIZE 128 + 32

#define ITTERATIONS 1
#define EPSILON 1e-2

int check (float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(a[i] - b[i]) > EPSILON) {
            printf("Error at index %d: %f != %f\n", i, a[i], b[i]);
            return -1;
        }
    }
    return 0;
}


void mv_squential(const float* a, const float* b, float* c, int m, int n) { 
    for (int i = 0; i < m; i++) {
        c[i] = 0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

//used for all the kernels with warp tiling
__inline__ __device__
float warpReduceSum(float val) {
    /*
    Using the shuffle warp level primitive to perform a warp-level reduction.
    This function will reduce the values in a warp to a single value.
    The __shfl_down_sync instructing shuffles the data stored in registers
    between threads in a warp. This reduces the need for memory accesses to 
    the shared memory.
    */
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__
void mv_reduce(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
    /*
    This kernel is a simple matrix-vector multiplication kernel.
    It uses the reduction pattern to compute the output vector.
    */
    
    extern __shared__ float sharedSum[];

    // Thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate row and column indices
    int col = tid % n;
    int row = tid / n;

    // Initialize the shared memory
    if (threadIdx.x < blockDim.x) {
        sharedSum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Bounds check for rows
    if (row >= m) return;

    // Compute partial product for this thread
    float localSum = (col < n) ? a[row * n + col] * b[col] : 0.0f;

    // Perform intra-block reduction
    sharedSum[threadIdx.x] = localSum;

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            sharedSum[threadIdx.x] += sharedSum[threadIdx.x + stride];
        }
    }

    // The final result from each block is stored in the global memory
    if (threadIdx.x == 0) {
        atomicAdd(&c[row], sharedSum[0]);
    }
}



__global__ 
void mv_warpReduceNoRollOver(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
    /*
    Matrix-Vector multiplication using warp-level reduction. 
    The architecture of this kernel designates a number of complete bocks to
    compute a single row of the output vector. (No thread rollover from one row to the next)
    The rows of the kernel are further divided into warps, each warp is responsible
    for computing a partial product of the row. The partial products are then summed
    using warp-level reduction.
    */

    extern __shared__ float warpSum[]; // Dynamic shared memory for warp sums

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rowLength = ((n + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int row = blockIdx.x / ((n + blockDim.x - 1) / blockDim.x);
    int col = tid % (rowLength);
    int lane = tid % 32;   // Lane index within the warp
    int warpIdx = (col / 32); // Warp index within the block
    int warpIdxLocal = (warpIdx % 32) % (blockDim.x / 32); // warp index within a block 0 - 31

    if (row > m) return; // Bounds check for rows
    if (tid < 32) warpSum[tid] = 0.0f;

    float localVal = 0.0f;
    localVal = (col < n) * a[row * n + col] * b[col];

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

        if (lane == 0) atomicAdd(&c[row], sum);
    }
}

__global__ 
void mv_warpReduceNoRollOverGranular(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
    /*
    This kernel is a modified version of the mv_warpReduceNoRollOver kernel.
    It uses the same architecture but computes multiple rows of the output vector.
    The number of rows computed is determined by the THREAD_GRAIN constant.
    The benifit of this kernel is that it reduces the number of memory accesses.
    Rather then loading in an element from the input vector for each element in 
    Matrix, the element is loaded once and used for multiple elements in the matrix.
    */

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
        }
    }
}

__global__ 
void mv_warpReduceNoRollOverGranularVectorized(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
    /*
    This kernel is a modified version of the mv_warpReduceNoRollOver kernel.
    It uses the same architecture but computes multiple rows of the output vector.
    The number of rows computed is determined by the THREAD_GRAIN constant.
    Vectorization is used to load in multiple elements from the input vector at once.
    It converts LDG.E.32 to LDG.E.128 to load in 4 elements at once.
    */

    extern __shared__ float warpSum[]; // Dynamic shared memory for warp sums

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int rowLength = (((n / THREAD_GRAIN) + blockDim.x - 1) / blockDim.x) * blockDim.x;
    int row = (blockIdx.x / ((n + blockDim.x - 1) / blockDim.x)) * THREAD_GRAIN;
    int col = tid % (rowLength);
    int lane = tid % 32;   // Lane index within the warp
    int warpIdx = (col / 32); // Warp index within the block
    int warpIdxLocal = (warpIdx % 32) % (blockDim.x / 32); // warp index within a block 0 - 31

    float bLocal[THREAD_GRAIN] = {0.0f};

    for (int phase = 0; phase < THREAD_GRAIN; phase++){
        if (row + phase > m) return; // Bounds check for rows

        float localVal = 0.0f;

        if (tid < 32) warpSum[tid] = 0.0f;

        if (col < n) {
            float4 a4 = reinterpret_cast<const float4*>(a)[(row + phase) + col];
            float4 b4 = reinterpret_cast<const float4*>(b)[col];
            localVal += a4.x * b4.x;
            localVal += (col + 1 < n) * a4.y * b4.y;
            localVal += (col + 2 < n) * a4.z * b4.z;
            localVal += (col + 3 < n) * a4.w * b4.w;
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
        }
    }
}

__global__
void mv_warpReduce(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
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
    // this is a safe assumption but without it we get the wrong results
    if (threadIdx.x < 64)  warpSum[threadIdx.x] = 0.0f;
    if (row >= m) return; // Bounds check for rows

    
    float localVal = 0.0f;
    // Load data from global memory
    // Using a bit operator to avoid branching (flag)
    localVal = (col < n) * a[row * n + col] * b[col];
    
    // Warp-level reduction
    localVal = warpReduceSum(localVal);

    // Store results in shared memory
    if (lane == 0) {
        // second row will always be stored starting at index 32
        warpSum[warpIdxLocal + warpIdyLocal * 32] = localVal;
    }

    // Synch is needed here because we need to ensure all the threads have written to shared memory
    __syncthreads();

    // Reduce warp results to a single value
    if (warpIdx == firstWarp || warpIdx == 0) {
        float sum = 0.0f;
        // warpIdx has a row associated with it, we can index the shared memory with the row with warpIdyLocal
        localVal = (lane < warpsPerRow) * warpSum[lane + warpIdyLocal * 32];
        sum = warpReduceSum(localVal);

        // Only the first warp in the block will write to the output vector
        if (lane == 0){ 
            atomicAdd(&c[row], sum);
        }
    }
}

__global__
void mv_warpReduceGranular(const float* __restrict__ a, const float* __restrict__ b, float* c, int m, int n) {
    /* 
    This kernel is a modified version of the mv_warpReduce kernel.
    It uses the same architecture but computes multiple rows of the output vector.
    The number of rows computed is determined by the THREAD_GRAIN constant.
    The benifit of this kernel is that it reduces the number of memory accesses.
    Rather then loading in an element from the input vector for each element in
    Matrix, the element is loaded once and used for multiple elements in the matrix.

    DOES NOT WORK RIGHT NOW
    */

    // designed for datasets with more then 1024 columns
    extern __shared__ float warpSum[]; // Dynamic shared memory for warp sums
    // The compliler will optimize these calculations and their register usage
    // Each thread can have up to 256 registers in the and this implemetation doesn't go above 10 registers per thread
    int warpsPerRow = (n + 31) / 32; // rounds up to the nearest multiple of 32
    int warpsPerBlock = (blockDim.x + 31) / 32; // rounds up to the nearest multiple of 32 

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int col = (tid % (warpsPerRow * 32));// Column index
    int row = ((tid / (warpsPerRow * 32)) * THREAD_GRAIN); // Row index (global row index)
    int lane = col % 32;   // Lane index within the warp

    int firstRow = (blockIdx.x * blockDim.x ) / (warpsPerRow * 32); // First row in the block
    int firstWarp = ((blockIdx.x * blockDim.x + 31) - row * warpsPerRow * 32) / 32; // First warp in the block

    int warpIdx = (col / 32) % warpsPerRow; // Warp index within the block
    int warpIdxLocal = (warpIdx % 32); // warp index within a block 0 - 31
    int warpIdy = row;  // Starts counting from zero for each block (local row index)
    int warpIdyLocal = warpIdy - firstRow ;

    float bLocal = b[col];

    for (int phase = 0; phase < THREAD_GRAIN; phase++){

        int sharedMemSize = 64;
        if (threadIdx.x < sharedMemSize) {
            warpSum[threadIdx.x] = 0.0f;
        }

        if (row + phase >= m) return; // Bounds check for rows

        float localVal = 0.0f;

        localVal += a[(row * n + phase * n) + col] * bLocal;
    
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

            localVal = (lane < warpsPerRow) * warpSum[lane + warpIdyLocal * 32];
            sum = warpReduceSum(localVal);
            
            if (lane == 0) {
                // warpIdx has a row associated with it
                atomicAdd(&c[row + phase], sum);
            }
        }
    }
}



//nvcc step2.cu -o step2 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64\cl.exe"
// -ccbin is the path to the cl.exe compiler

int main() {
    const int M = 256; // Number of rows
    const int N = 2700; // Number of columns
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * sizeof(float);
    size_t sizeC = M * sizeof(float);

    float *h_a = (float*)malloc(sizeA);
    float *h_b = (float*)malloc(sizeB);
    float *h_c = (float*)malloc(sizeC);
    float *h_d = (float*)malloc(sizeC);
    float *h_c2 = (float*)malloc(sizeC);
    float *h_c3 = (float*)malloc(sizeC);
    float *h_c4 = (float*)malloc(sizeC);

    float *seq_c = (float*)malloc(sizeC);

    // Initialize host arrays
    for (int i = 0; i < M * N; i++) {
        h_a[i] = static_cast<float>(rand() % 2);
    }
    for (int i = 0; i < N; i++) {
        h_b[i] = static_cast<float>(rand() % 3);
    }
    for (int i = 0; i < M; i++) {
        h_d[i] = static_cast<float>(rand() % 3);
    }

    float *d_a, *d_b, *d_d;
    float *d_c, *d_c2, *d_c3, *d_c4;
    
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_d, sizeC);
    cudaMalloc(&d_c, sizeC);
    cudaMalloc(&d_c2, sizeC);
    cudaMalloc(&d_c3, sizeC);
    cudaMalloc(&d_c4, sizeC);

    cudaMemcpy(d_a, h_a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, sizeC, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, sizeC);
    cudaMemset(d_c2, 0, sizeC);
    cudaMemset(d_c3, 0, sizeC);
    cudaMemset(d_c4, 0, sizeC);

    dim3 block(BLOCK_SIZE); // Threads per block

    // Warp tiling needs special grid size calculations
    // A warp cannot span multiple rows, so we need to 
    // calculate the number of warps per row and the number
    int warpsPerRow = (N + 31) / 32;
    int warpsPerBlock = (BLOCK_SIZE + 31) / 32;

    // M is the number of rows
    // (warpsPerRow + warpsPerBlock - 1) / warpsPerBlock is the number of blocks needed to cover a row
    // This will round up to the nearest block to ensure all rows are covered
    dim3 grid_mvWarpReduceNoRoll((M * warpsPerRow + warpsPerBlock - 1) / warpsPerBlock); // Blocks in grid, mv_warpReduceNoRollOver

    // This grid size is used for the granular version of the kernel
    // It is the same as the previous grid size but with the THREAD_GRAIN constant
    // M + THREAD_GRAIN - 1 / THREAD_GRAIN is the number of rows computed by a single block
    // This will round up to the nearest row to ensure all rows are covered
    dim3 gridGranular((((M + THREAD_GRAIN -1) / THREAD_GRAIN) * warpsPerRow + warpsPerBlock - 1) / warpsPerBlock); // Blocks in grid, mv_warpReduceNoRollOverGranular

    int warpsPerRowVectorized = (N + 127) / 128;
    int warpsPerBlockVectorized = (BLOCK_SIZE + 127) / 128;

    dim3 gridVectorized((((M + THREAD_GRAIN -1) / THREAD_GRAIN) * warpsPerRowVectorized + warpsPerBlockVectorized - 1) / warpsPerBlockVectorized); // Blocks in grid, mv_warpReduceNoRollOverGranularVectorized

    dim3 grid_mvReduce(M * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)); // Blocks in grid, mv_reduce

    dim3 grid_mvWarpReduce((M * warpsPerRow + warpsPerBlock - 1) / warpsPerBlock); // Blocks in grid, mv_WarpReduce

    size_t sharedMemorySize = 64 * sizeof(float);//(BLOCK_SIZE + 31 / 32) * sizeof(float); // Shared memory for warp sums

    //timers for each kernel
    float time[5];
    float time_avg[5] = {0.0f};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sequential
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        mv_squential(h_a, h_b, seq_c, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[0], start, stop);
        time_avg[0] += time[0];
    }
    time_avg[0] /= ITTERATIONS;

    // mv_reduce
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        mv_reduce<<<grid_mvReduce, block, sharedMemorySize>>>(d_a, d_b, d_c, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[1], start, stop);
        time_avg[1] += time[1];
    }
    time_avg[1] /= ITTERATIONS;

    cudaMemcpy(h_c, d_c, sizeC, cudaMemcpyDeviceToHost);

    // mv_warpReduce
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        mv_warpReduce<<<grid_mvWarpReduce, block, sharedMemorySize>>>(d_a, d_b, d_c2, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[2], start, stop);
        time_avg[2] += time[2];
    }
    time_avg[2] /= ITTERATIONS;

    cudaMemcpy(h_c2, d_c2, sizeC, cudaMemcpyDeviceToHost);

    // mv_warpReduceNoRollOver
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        mv_warpReduceNoRollOver<<<grid_mvWarpReduceNoRoll, block, sharedMemorySize>>>(d_a, d_b, d_c3, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[3], start, stop);
        time_avg[3] += time[3];
    }
    time_avg[3] /= ITTERATIONS;

    cudaMemcpy(h_c3, d_c3, sizeC, cudaMemcpyDeviceToHost);

    // mv_warpReduceNoRollOverGranular
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        mv_warpReduceNoRollOverGranular<<<gridGranular, block, sharedMemorySize>>>(d_a, d_b, d_c3, M, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[4], start, stop);
        time_avg[4] += time[4];
    }
    time_avg[4] /= ITTERATIONS;

    cudaMemcpy(h_c4, d_c4, sizeC, cudaMemcpyDeviceToHost);

    printf("Sequential: %f ms\n", time_avg[0]);
    printf("mv_reduce: %f ms\n", time_avg[1]);
    printf("mv_warpReduce: %f ms\n", time_avg[2]);
    printf("mv_warpReduceNoRollOver: %f ms\n", time_avg[3]);
    printf("mv_warpReduceNoRollOverGranular: %f ms\n", time_avg[4]);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    free(seq_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


__global__
void vva(float *a, float *b, float *c, int n) {
    /*
        This kernel is used for vector vector addition.
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}