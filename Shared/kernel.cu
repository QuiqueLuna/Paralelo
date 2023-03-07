
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;


__global__ void transpose(int* a, int* b, int n) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;


    int row = gid / n;
    int col = gid - row * n;
    if (gid < n * n) {
        s[row * n + col] = a[row * n + col];
        __syncthreads();
        b[col * n + row] = s[row * n + col];
    }
}

__global__ void conv(int* a, int* b, int n) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;


    int row = gid / n;
    int col = gid - row * n;

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
    dim3 grid(32 / (n * n), 32 / (n * n));
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


/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

void bubble_sort(int *a, int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (a[j] > a[j+1]) {
                int aux = a[j+1];
                a[j+1] = a[j];
                a[j] = aux;
            }
        }
    }
}

__global__ void bubbleSortDev(int *a, int n) {
    __shared__ int s[64];

    int index = threadIdx.x;


    __syncthreads();
}

int main(){
    int size = 10;
    int* host_a, *res;
    int* dev_a;
    
    host_a = (int*)malloc(size*sizeof(int));
    res = (int*)malloc(size*sizeof(int));
    cudaMalloc(&dev_a, size * sizeof(size));


    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        host_a[i] = r1; 
        printf("%d ", host_a[i]);
    }
    printf("\n");


    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid(1);
    dim3 block(size);
    bubbleSortDev << <grid, block >> > (dev_a,size);
    cudaMemcpy(res, dev_a, size * sizeof(int), cudaMemcpyHostToDevice);



    bubble_sort(host_a,size);
    printf("CPU: \n");
    for (int i = 0; i < size; i++) {
        printf("%d ", host_a[i]);
    }
    printf("\n");

    printf("GPU\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");

    return 0;
}

*/