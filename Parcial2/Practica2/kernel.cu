#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>


void bubble_sort(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int aux = a[j + 1];
                a[j + 1] = a[j];
                a[j] = aux;
            }
        }
    }
}
__global__ void bubbleSortDev(int* a, int n) {
    int tid = threadIdx.x;

    for (int i = 0; i < n; i++) {

        int offset = i % 2;
        int leftIndex = 2 * tid + offset;
        int rightIndex = leftIndex + 1;

        if (rightIndex < n) {
            if (a[leftIndex]>a[rightIndex]) {
                int aux = a[leftIndex];
                a[leftIndex] = a[rightIndex];
                a[rightIndex] = aux;
            }
        }
        __syncthreads();
    }
}
int main() {
    int size = 10;
    int* host_a, * res;
    int* dev_a;
    host_a = (int*)malloc(size * sizeof(int));
    res = (int*)malloc(size * sizeof(int));
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
    bubbleSortDev << <grid, block >> > (dev_a, size);
    cudaMemcpy(res, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    bubble_sort(host_a, size);
    
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