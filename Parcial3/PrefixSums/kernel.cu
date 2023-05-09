#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void prefSum(int* a, int* b, int n) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < n) {
        for (int i = tid; i < n; i++) {
            b[i] += a[tid];
            __syncthreads();
        }

    }
}
/*
1 1 1 1 1 1
1 2 2 2 2 2
1 2 3 4 4 4
1 2 3 5

1 1 1 1 1 1
1 2 1 2 1 2



1 2 3 4 5 6
*/

int main() {
    int size = 1024;
    int* host_a, *host_b;
    int* dev_a, * dev_b;
    
    host_a = (int*)malloc(size * sizeof(int));
    host_b = (int*)malloc(size * sizeof(int));
    cudaMalloc(&dev_a, size * sizeof(size));
    cudaMalloc(&dev_b, size * sizeof(size));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (10));
        host_a[i] = 1;
        host_b[i] = 0;
        printf("%d ", host_a[i]);
    }
    printf("\n");

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(32);
    dim3 block(64);
    prefSum << <grid, block >> > (dev_a, dev_b, size);
    cudaMemcpy(host_b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);

    printf("RES\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", host_b[i]);
    }
    printf("\n");
    return 0;
}