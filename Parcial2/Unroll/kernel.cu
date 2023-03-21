
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

int main() {

    const int n = 12;
    int* host_a, * host_b;
    int* dev_a, * dev_b;
    host_a = (int*)malloc(n * sizeof(int));
    host_b = (int*)malloc(n * sizeof(int));

    cudaMalloc(&dev_a, n * sizeof(int));
    cudaMalloc(&dev_b, n * sizeof(int));


    for (int i = 0; i < n; i++) {
        int r1 = (rand() % (5));
        host_a[i] = r1;
    }


    ///
    printf("A: \n");
    for (int i = 0; i < n; i++) {
        printf("%d ", host_a[i]);
    }
    printf("\n");
    ///

    cudaMemcpy(dev_a, host_a, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(32 / n, 32 / n);
    unroll << <grid, block >> > (dev_a, dev_b, n);
    cudaMemcpy(host_b, dev_b, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("B: \n");
    for (int i = 0; i < n; i++) {
            cout << host_b[i] << " ";
        }
    printf("\n");
    

    cudaDeviceSynchronize();
    cudaDeviceReset();

    free(host_a);
    free(host_b);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}