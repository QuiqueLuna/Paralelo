
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void mult(int *a, int *b, int *c){
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int main()
{
    const int N = 3;
    int size = N * sizeof(N);
    const int a[N] = {1, 0, 1};
    const int b[N] = {2, 4, 3};
    int c[N] = {0, 0, 0};

    int* d_a = 0;
    int* d_b = 0;
    int* d_c = 0;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,c,size,cudaMemcpyHostToDevice);

    mult << <1, N >> > (d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost);
    

    printf("{1,0,1} * {2,4,3} = {%d,%d,%d}\n",c[0], c[1], c[2]);

    cudaDeviceReset();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
