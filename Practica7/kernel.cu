
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;


__global__ void transpose(int* a, int* b, int n) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;


    int row = gid / n;
    int col = gid - row * n;
    if (gid < n * n) {
        b[col * n + row] = a[row * n + col];
    }
}

int main() {

    const int n = 4;
    int* host_a, * host_b;
    int* dev_a, * dev_b;
    host_a = (int*)malloc(n * n * sizeof(int));
    host_b = (int*)malloc(n * n * sizeof(int));
    cudaMalloc(&dev_a, n * n * sizeof(int));
    cudaMalloc(&dev_b, n * n * sizeof(int));

    for (int i = 0; i < n * n; i++) {
        int r1 = (rand() % (256));
        host_a[i] = r1;
        host_b[i] = 0;
    }

    ///
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", host_a[i * n + j]);
        }
        printf("\n");
    }
    ///

    cudaMemcpy(dev_a, host_a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(32/(n*n), 32/(n*n));
    transpose << <1, block >> > (dev_a, dev_b, n);
    cudaMemcpy(host_b, dev_b, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    cout << "Res:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << host_b[i * n + j] << " ";
        }
        cout << "\n";
    }
    free(host_a);
    free(host_b);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}