
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void unroll(int* a, int* b, int n) {
    int tid = threadIdx.x;
    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
    int index = BLOCK_OFFSET + tid;

    int* i_data = a + BLOCK_OFFSET;

    if ((index + blockDim.x) < n) {
        a[index] += a[index + blockDim.x];
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            i_data += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        b[blockIdx.x] = i_data[0];
    }
}

__global__ void unrolling_complete(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int* i_data = int_array + blockDim.x * blockIdx.x;

    if (blockDim.x == 1024 && tid < 512)
        i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if (blockDim.x == 512 && tid < 256)
        i_data[tid] += i_data[tid + 256];
    __syncthreads;

    if (blockDim.x == 256 && tid < 128)
        i_data[tid] += i_data[tid + 128];
    __syncthreads();
    
    if (blockDim.x == 128 && tid < 64)
        i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
    }
            
}

int main() {

    //const int n = 256;
    const int n = 1024;
    int* host_a, * host_b;
    int* dev_a, * dev_b;
    host_a = (int*)malloc(n * sizeof(int));
    host_b = (int*)malloc(n * sizeof(int));

    cudaMalloc(&dev_a, n * sizeof(int));
    cudaMalloc(&dev_b, n * sizeof(int));


    for (int i = 0; i < n; i++) {
        int r1 = (rand() % (5));
        //host_a[i] = r1;
        host_a[i] = 1;
    }


    ///
    printf("A: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", host_a[i]);
    }
    printf("\n");
    ///

    cudaMemcpy(dev_a, host_a, n * sizeof(int), cudaMemcpyHostToDevice);

    //dim3 block(128);
    //dim3 grid(256 / 128 / 2);
    //unroll << <grid, block >> > (dev_a, dev_b, n);
   
    dim3 block(128);
    dim3 grid(8);
    unrolling_complete << <grid, block >> > (dev_a, dev_b, n);
    cudaMemcpy(host_a, dev_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, dev_b, n * sizeof(int), cudaMemcpyDeviceToHost);


    printf("A: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", host_a[i]);
    }
    printf("\n");

    printf("B: \n");
    for (int i = 0; i < n; i++) {
        cout << host_b[i] << " ";
    }
    printf("\n");
    
    int res = 0;
    printf("RES: \n");
    for (int i = 0; i < grid.x; i++) {
        res+=host_b[i];
    }
    printf("%d\n", res);


    cudaDeviceSynchronize();
    cudaDeviceReset();

    free(host_a);
    free(host_b);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}