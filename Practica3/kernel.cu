
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*
__global__ void idx_calc_tid(int *input)
{
    int tid = threadIdx.x;
    printf("[DEVICE] threadIdx.x %d, data: %d\n\r", tid, input[tid]);
}
*/

/*
__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;
    printf("[DEVICE] blockIdx.x %d, threadIdx.x %d, gid: %d, data: %d\n\r", blockIdx.x, tid, gid, input[gid]);
}
*/

/*
__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x;
    int offsetBlock = blockIdx.x * blockDim.x;
    int offsetRow = blockIdx.y * blockDim.x * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;
    printf("[DEVICE] gridDim.x %d,  blockIdx.x %d, blockIdz.y %d, threadIdx.x %d, gid: %d, data: %d\n\r", gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}
*/

__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;
    printf("[DEVICE] gridDim.x %d,  blockIdx.x %d, blockIdz.y %d, threadIdx %d, gid: %d, data: %d\n\r", gridDim.x, blockIdx.x, blockIdx.y, tid, gid, input[gid]);
}

int main(){
    /*
    const int N = 16;
    int size = N * sizeof(int);
    const int a[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    int* d_a;

    cudaMalloc(&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    idx_calc_tid << <1, N >> > (d_a);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);
    */

    /*
    const int N = 16;
    int size = N * sizeof(int);
    const int a[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

    int* d_a;

    cudaMalloc(&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    idx_calc_tid << <2, 8 >> > (d_a);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);
    */
    
    /*
    const int N = 16;
    int size = N * sizeof(int);
    const int a[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

    int* d_a;

    cudaMalloc(&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    idx_calc_tid << <4, 4 >> > (d_a);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);
    */

    /*
    const int N = 16;
    int size = N * sizeof(int);
    const int a[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

    int* d_a;

    cudaMalloc(&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    dim3 grid(2,2);
    dim3 block(4);
    idx_calc_tid << <grid, block >> > (d_a);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);
    */

    const int N = 16;
    int size = N * sizeof(int);
    const int a[N] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

    int* d_a;

    cudaMalloc(&d_a, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    dim3 grid(2,2);
    dim3 block(2,2);
    idx_calc_tid << <grid, block >> > (d_a);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    cudaFree(d_a);

    return 0;
}

