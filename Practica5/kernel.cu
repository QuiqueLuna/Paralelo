
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort)exit(code);
    }
}

/*
  int d_Count = 1;

    for (int devNo = 0; devNo < d_Count; devNo++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devNo);
        printf(" Device Number: %d\n", devNo);
        printf(" Device name: %s\n", prop.name);
        printf(" No. of MultiProcessors: %d\n", prop.multiProcessorCount);
        printf(" Compute Capability: %d, %d\n", prop.major, prop.minor);
        printf(" Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf(" Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf(" Peak Memory Bandwidth (GB/s) : %8.2f\n",  2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf(" Total amount of Global Memory : %dKB \n", prop.totalGlobalMem / 1024);
        printf(" Total amount of Const Memory: %dKB\n", prop.totalConstMem / 1024);
        printf(" Total of Shared Memory per block : %dKB\n", prop.sharedMemPerBlock / 1024);
        printf(" Total of Shared Memory per MP: %dKB\n", prop. sharedMemPerBlock / 1024) ;
        printf(" Warp Size: %d\n", prop.warpSize);
        printf(" Max. threds per block: %d\n", prop.maxThreadsPerBlock);
        printf(" Max.threds per MP : %d\n", prop.maxThreadsPerMultiProcessor);
        printf(" Maximum number of warps per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor / 32);
        printf(" Maximum Grid size: (%d, %d, %d) \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf(" Maximum block dimension: (%d, %d , %d) \n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
    }
*/

__global__ void matrix_mult(int *a, int *b, int *c, int s) {
    int row = threadIdx.x/s;
    int col = threadIdx.x - row*s;

    int suma = 0;
    if (row < s && col < s) {
        for (int i = 0; i < s; i++) {
            suma += a[row * s + i] * b[i * s + col];
        }
    }
    c[threadIdx.x] = suma;
}

int main(){
    
    const int side = 2;
    int* host_a, * host_b, * host_c;
    int* dev_a, * dev_b, * dev_c;
    host_a = (int*)malloc(side*side * sizeof(int));
    host_b = (int*)malloc(side * side * sizeof(int));
    host_c = (int*)malloc(side * side * sizeof(int));
    cudaMalloc(&dev_a, side * side * sizeof(int));
    cudaMalloc(&dev_b, side * side * sizeof(int));
    cudaMalloc(&dev_c, side * side * sizeof(int));
    for (int i = 0; i < side * side; i++) {
        int r1 = (rand() % (256));
        int r2 = (rand() % (256));
        host_a[i] = r1;
        host_b[i] = r2;
        host_c[i] = 0;
    }
    cudaMemcpy(dev_a, host_a, side * side * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, side * side * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, side * side * sizeof(int), cudaMemcpyHostToDevice);

    matrix_mult << <1, 32 >> > (dev_a, dev_b, dev_c, side);
    cudaMemcpy(host_c, dev_c, side* side * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cout << "A:\n";
    for (int i = 0; i < side; i++) {
        for (int j = 0; j < side; j++) {
            cout << host_a[i * side + j] << " ";
        }
        cout << "\n";
    }
    cout << "B:\n";
    for (int i = 0; i < side; i++) {
        for (int j = 0; j < side; j++) {
            cout << host_b[i * side + j] << " ";
        }
        cout << "\n";
    }

    cout << "C: \n";
    for (int i = 0; i < side; i++) {
        for (int j = 0; j < side; j++) {
            cout << host_c[i * side + j] << " ";
        }
        cout << "\n";
    }
    free(host_a);
    free(host_b);
    free(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    return 0;
}