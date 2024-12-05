
    
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
    
    

    #difine BLOCK_SIZE 256
    #define THREAD_GRAIN 4

    dim3 block(BLOCK_SIZE); // Threads per block
    dim3 grid((M / THREAD_GRAIN) * ((N + BLOCK_SIZE - 1) / BLOCK_SIZE)); // Blocks in grid
    dim3 blockvva(BLOCK_SIZE); // Threads per block
    dim3 gridvva((M + BLOCK_SIZE - 1) / BLOCK_SIZE); // Blocks in grid

    mv_warpReduceNoRollOverGranular<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(M_G, w_v, zhat_vp, M, N);
    vva<<<gridvva, blockvva>>>(zhat_v, zhat_vp, g_P, M);

