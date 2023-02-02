#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/*
__global__ void idx_calc_tid(){

    int tid = threadIdx.z * blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * blockDim.z * gridDim.x;

    int offsetLayer = blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;
    int gid = tid + offsetBlock + offsetRow + offsetLayer;

    printf("[DEVICE]gid: %d\n\r", gid);
}
*/

__global__ void sum_array_gpu(int *a, int *b, int *c, int *d, int size) {

    int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * blockDim.z * gridDim.x;

    int offsetLayer = blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y;
    int gid = tid + offsetBlock + offsetRow + offsetLayer;

    if (gid < size) {
        d[gid] = a[gid] + b[gid] + c[gid];
        //printf("[DEVICE]: %d = %d + %d\n\r", c[gid], a[gid], b[gid]);

    }

    //printf("[DEVICE]gid: %d\n\r", gid);
}

void sum_array_cpu(int* a, int* b, int* c, int* d, int size) {
    for (int i = 0; i < size; i++) {
        d[i] = a[i] + b[i] + c[i];
    }
}

int main(){
    /*
    dim3 grid(2, 2, 2);
    dim3 block(2, 2, 2);
    idx_calc_tid << <grid, block >> > ();

    cudaDeviceSynchronize();
    cudaDeviceReset();
    */

    /*
    const int size = 10000;
    int* host_a, * host_b, * host_c, *check;
    int* dev_a, * dev_b, * dev_c;

    host_a = (int*)malloc(size * sizeof(int));
    host_b = (int*)malloc(size * sizeof(int));
    host_c = (int*)malloc(size * sizeof(int));
    check = (int*)malloc(size * sizeof(int));

    cudaMalloc(&dev_a, size*sizeof(int));
    cudaMalloc(&dev_b, size*sizeof(int));
    cudaMalloc(&dev_c, size*sizeof(int));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        int r2 = (rand() % (256));

        host_a[i] = r1;
        host_b[i] = r2;
        host_c[i] = 0;
    }

    cudaMemcpy(dev_a,host_a,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,host_b,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c,host_c,size*sizeof(int),cudaMemcpyHostToDevice);

    sum_array_cpu(host_a,host_b,host_c,size);
    sum_array_gpu << <79, 128 >> > (dev_a,dev_b,dev_c,size);
    cudaMemcpy(check,dev_c,size*sizeof(int),cudaMemcpyDeviceToHost);
    //cudaDeviceSynchronize();
    //cudaDeviceReset();

    bool ok = true;
    for (int i = 0; i < size; i++) {
        if (host_c[i] != check[i]) {
            ok = false;
        }
    }
    
    
    printf("Host\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", host_c[i]);
    }
    printf("\n");
    
    printf("Device\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", check[i]);
    }
    printf("\n");
    

    if (ok) {
        printf("Equal\n");
    }
    else {
        printf("Not Equal\n");
    }
    free(host_a);
    free(host_b);
    free(host_c);
    free(check);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    */


    const int size = 10000;
    int* host_a, * host_b, * host_c, *host_d, * check;
    int* dev_a, * dev_b, * dev_c, *dev_d;

    host_a = (int*)malloc(size * sizeof(int));
    host_b = (int*)malloc(size * sizeof(int));
    host_c = (int*)malloc(size * sizeof(int));
    host_d = (int*)malloc(size * sizeof(int));
    check = (int*)malloc(size * sizeof(int));

    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_b, size * sizeof(int));
    cudaMalloc(&dev_c, size * sizeof(int));
    cudaMalloc(&dev_d, size * sizeof(int));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        int r2 = (rand() % (256));
        int r3 = (rand() % (256));

        host_a[i] = r1;
        host_b[i] = r2;
        host_c[i] = r3;
        host_d[i] = 0;
    }

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d, host_d, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(8,4,4);
    dim3 block(8,4,4);

    sum_array_cpu(host_a, host_b, host_c, host_d, size);
    sum_array_gpu << <grid, block >> > (dev_a, dev_b, dev_c, dev_d, size);
    cudaMemcpy(check, dev_d, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    bool ok = true;
    for (int i = 0; i < size; i++) {
        if (host_d[i] != check[i]) {
            ok = false;
        }
    }


    printf("Host\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", host_d[i]);
    }
    printf("\n");

    printf("Device\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", check[i]);
    }
    printf("\n");


    if (ok) {
        printf("Equal\n");
    }
    else {
        printf("Not Equal\n");
    }
    free(host_a);
    free(host_b);
    free(host_c);
    free(check);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    return 0;
}