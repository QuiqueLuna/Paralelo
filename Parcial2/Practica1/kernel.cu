
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

__global__ void conv(int* a, int* b, int* k, int n, int m, int kernelSize) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;


    int row = gid / n;
    int col = gid - row * n;

    int suma = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (row + i >= 0 && row + i < n && col + j>=0 && col + j < n) {
                s[(i + 1) * kernelSize + j + 1] = k[(i + 1) * kernelSize + j + 1];
                __syncthreads();
                suma += a[(row + i) * m + col + j] * s[(i + 1) * kernelSize + j + 1];
                //suma += a[(row + i) * m + col + j] * k[(i + 1) * kernelSize + j + 1];

            }
        }
    }
    //printf("row: %d col: %d suma: %d\n", row, col, suma);
    b[row * m + col] = suma;

}

int main() {

    const int nKernel = 3, n=8,m=8;
    int* host_aKernel, * host_bKernel, *host_a, *host_b;
    int* dev_aKernel, * dev_bKernel, *dev_a, *dev_b;
    host_aKernel = (int*)malloc(nKernel * nKernel * sizeof(int));
    host_bKernel = (int*)malloc(nKernel * nKernel * sizeof(int));
    host_a = (int*)malloc(n * m * sizeof(int));
    host_b = (int*)malloc(n * m * sizeof(int));

    cudaMalloc(&dev_aKernel, nKernel * nKernel * sizeof(int));
    cudaMalloc(&dev_bKernel, nKernel * nKernel * sizeof(int));
    cudaMalloc(&dev_a, n * m * sizeof(int));
    cudaMalloc(&dev_b, n * m * sizeof(int));


    for (int i = 0; i < nKernel * nKernel; i++) {
        int r1 = (rand() % (1));
        host_aKernel[i] = r1;
        host_bKernel[i] = 0;
    }

    for (int i = 0; i < n * m; i++) {
        int r1 = (rand() % (3));
        host_a[i] = r1;
        host_b[i] = 0;
    }

    ///
    host_aKernel[3] = 1;
    printf("Kernel: \n");
    for (int i = 0; i < nKernel; i++) {
        for (int j = 0; j < nKernel; j++) {
            printf("%d ", host_aKernel[i * nKernel + j]);
        }
        printf("\n");
    }

    printf("A: \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", host_a[i * m + j]);
        }
        printf("\n");
    }
    ///

    cudaMemcpy(dev_aKernel, host_aKernel, nKernel * nKernel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bKernel, host_bKernel, nKernel * nKernel * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a, host_a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, n * m * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid(32 / (nKernel * nKernel), 32 / (nKernel * nKernel));
    transpose << <grid, block >> > (dev_aKernel, dev_bKernel, nKernel);
    cudaMemcpy(host_bKernel, dev_bKernel, nKernel * nKernel * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Res Kernel:\n";
    for (int i = 0; i < nKernel; i++) {
        for (int j = 0; j < nKernel; j++) {
            cout << host_bKernel[i * nKernel + j] << " ";
        }
        cout << "\n";
    }

    dim3 block2(32, 32);
    dim3 grid2((64+ (n * m) -1)/(n * m), (64 + (n * m) - 1) /(n * m));
    conv << <grid2, block2 >> > (dev_a,dev_b,dev_bKernel,n,m,nKernel);
    cudaMemcpy(host_b, dev_b, n * m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();


    printf("B: \n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", host_b[i * m + j]);
        }
        printf("\n");
    }

    free(host_aKernel);
    free(host_bKernel);
    cudaFree(dev_aKernel);
    cudaFree(dev_bKernel);

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