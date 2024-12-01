#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <cstdio>

#define THREAD_GRAIN 4
#define BLOCK_SIZE 256
#define ITTERATIONS 500

void seq_dp(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[0] += a[i] * b[i];
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
void dp_warpReduce(const float* __restrict__ a, const float* __restrict__ b, float* c, int n) {
    extern __shared__ float warpSum[]; // Dynamic shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % 32;
    int warpId = threadIdx.x / 32; // since threadIdx.x is block level two warps in different blocks can have the same warpId
    int maxWarpIdBlock = (blockDim.x + 31) / 32;

    float localVal = 0.0f;

    // Compute dot product for THREAD_GRAIN elements
    // Coalesced access 
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        localVal += (i < n) * a[i] * b[i];
    }

    // Warp-level reduction
    localVal = warpReduceSum(localVal);

    // Store warp results in shared memory
    if (lane == 0) {
        warpSum[warpId] = localVal;
    }
    __syncthreads();

    // Final block-level reduction by the first warp
    if (warpId == 0) {
        localVal = (lane < maxWarpIdBlock) * warpSum[lane];
        localVal = warpReduceSum(localVal);

        if (lane == 0) {
            atomicAdd(c, localVal);
        }
    }
}

__global__ 
void dp_warpReduce_granular(const float* __restrict__ a, const float* __restrict__ b, float* c, int n) {
    extern __shared__ float warpSum[]; // Dynamic shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % 32;
    int warpId = threadIdx.x / 32; // since threadIdx.x is block level two warps in different blocks can have the same warpId
    int maxWarpIdBlock = (blockDim.x + 31) / 32;

    float localVal = 0.0f;

    // Compute dot product for THREAD_GRAIN elements
    // Coalesced access 
    #pragma unroll
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {    
        localVal += (i < n) * a[i] * b[i];
    }

    // Warp-level reduction
    localVal = warpReduceSum(localVal);

    // Store warp results in shared memory
    if (lane == 0) {
        warpSum[warpId] = localVal;
    }
    __syncthreads();

    // Final block-level reduction by the first warp
    if (warpId == 0) {
        localVal = (lane < maxWarpIdBlock) ? warpSum[lane] : 0.0f;
        localVal = warpReduceSum(localVal);

        if (lane == 0) {
            atomicAdd(c, localVal);
        }
    }
}

__global__
void dp_warpReduceVector(const float* __restrict__ a, const float* __restrict__ b, float* c, int n) {
    extern __shared__ float warpSum[]; // Dynamic shared memory
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % 32;
    int warpId = threadIdx.x / 32;
    int maxWarpIdBlock = (blockDim.x + 31) / 32;

    float localVal = 0.0f;

    int index = tid * 4;
    if (index + 3 < n) {
        float4 a_val = reinterpret_cast<const float4*>(&a[index])[0];
        float4 b_val = reinterpret_cast<const float4*>(&b[index])[0];
        localVal += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
    }

    // Warp-level reduction
    localVal = warpReduceSum(localVal);

    // Store warp results in shared memory
    if (lane == 0) {
        warpSum[warpId] = localVal;
    }
    __syncthreads();

    // Final block-level reduction by the first warp
    if (warpId == 0) {
        localVal = (lane < maxWarpIdBlock) * warpSum[lane];
        localVal = warpReduceSum(localVal);

        if (lane == 0) {
            atomicAdd(c, localVal);
        }
    }
}

int main() {
    int n = 60 * 30 * 3660;
    float *a, *b, *c, *c1, *c2, *c3;
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    c = (float*)malloc(sizeof(float));
    c1 = (float*)malloc(sizeof(float));
    c2 = (float*)malloc(sizeof(float));
    c3 = (float*)malloc(sizeof(float));

    float *d_a, *d_b, *d_c, *d_c2, *d_c3;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, sizeof(float));
    cudaMalloc(&d_c2, sizeof(float));
    cudaMalloc(&d_c3, sizeof(float));

    c[0] = c1[0] = c2[0] = c3[0] = 0.0f;

    srand(time(0));
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(rand() % 3);
        b[i] = static_cast<float>(rand() % 2);
    }

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, sizeof(float));
    cudaMemset(d_c2, 0, sizeof(float));
    cudaMemset(d_c3, 0, sizeof(float));


    //timers for each kernel
    float time[4];
    float time_avg[4] = {0.0f, 0.0f, 0.0f};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Sequential
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        seq_dp(a, b, c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[0], start, stop);
        time_avg[0] += time[0];
    }
    time_avg[0] /= ITTERATIONS;

    // Kernel 1
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        dp_warpReduce<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[1], start, stop);
        time_avg[1] += time[1];
    }
    time_avg[1] /= ITTERATIONS;

    // Kernel 2
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        dp_warpReduce_granular<<<(n + BLOCK_SIZE - 1) / (BLOCK_SIZE * THREAD_GRAIN), BLOCK_SIZE, sizeof(float) * BLOCK_SIZE>>>(d_a, d_b, d_c2, n);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[2], start, stop);
        time_avg[2] += time[2];
    }
    time_avg[2] /= ITTERATIONS;

    // Kernel 3
    for (int i = 0; i < ITTERATIONS; i++){
        cudaEventRecord(start);
        dp_warpReduceVector<<<(n + BLOCK_SIZE - 1) / (BLOCK_SIZE * 4), BLOCK_SIZE, sizeof(float) * BLOCK_SIZE>>>(d_a, d_b, d_c3, n);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time[3], start, stop);
        time_avg[3] += time[3];
    }
    time_avg[3] /= ITTERATIONS;

    cudaMemcpy(c1, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, d_c2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c3, d_c3, sizeof(float), cudaMemcpyDeviceToHost);

    //kernel dimension launch sizes
    printf("Kernel 1: Block size: %d, Grid size: %d\n", BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("Kernel 2: Block size: %d, Grid size: %d\n", BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("Kernel 3: Block size: %d, Grid size: %d\n", BLOCK_SIZE, (n + BLOCK_SIZE - 1) / (BLOCK_SIZE * THREAD_GRAIN));

    printf("\n");
    printf("Sequential: %f\n", c[0]);
    printf("Kernel 1: %f\n", c1[0]);
    printf("Kernel 2: %f\n", c2[0]);
    printf("Kernel 3: %f\n", c3[0]);

    printf("\n");
    printf("Time taken for sequential: %f ms\n", time_avg[0]);
    printf("Time taken for kernel 1: %f ms\n", time_avg[1]);
    printf("Time taken for kernel 2: %f ms\n", time_avg[2]);
    printf("Time taken for kernel 3: %f ms\n", time_avg[3]);

    printf("\n");
    printf("Speedup of kernel 1: %f\n", time_avg[0] / time_avg[1]);
    printf("Speedup of kernel 2: %f\n", time_avg[0] / time_avg[2]);
    printf("Speedup of kernel 3: %f\n", time_avg[0] / time_avg[3]);

    printf("\n");
    printf("Speedup of kernel 2 over kernel 3: %f\n", time_avg[2] / time_avg[3]);
    printf("Percent Speedup of kernel 2 over kernel 3: %f\n", (time_avg[2] - time_avg[3]) / time_avg[3] * 100);
    

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_c2); cudaFree(d_c3); 
    free(a); free(b); free(c); free(c1); free(c2); free(c3); 

    return 0;
}
