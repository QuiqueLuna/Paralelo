#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define LEN 1<<4

struct testStruct {
    int x;
    int y;
};

struct structArray {
    int x[LEN];
    int y[LEN];
};

__global__ void test_aos(testStruct* in, testStruct* result, const int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        testStruct temp = in[gid];
        temp.x += 5;
        temp.y += 10;
        result[gid] = temp;
    }
}

__global__ void test_soa(structArray* data, structArray* result, const int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        int tmpx = data->x[gid];
        int tmpy = data->y[gid];

        tmpx += 4;
        tmpy += 8;
        result->x[gid] = tmpx;
        result->y[gid] = tmpy;
    }
}

int main() {
    /*
    //Test AOS
    printf("AOS\n");
    int array_size = LEN;
    int byte_size = sizeof(testStruct) * array_size;
    int block_size = 32;

    testStruct* h_in, * h_res;
    h_in = (testStruct*)malloc(byte_size);
    h_res = (testStruct*)malloc(byte_size);

    for (int i = 0; i < array_size; i++) {
        h_in[i].x = 1;
        h_in[i].y = 2;
    }
    
    printf("OG: \n");
    for (int i = 0; i < array_size; i++) {
        printf("x: %d y: %d\n", h_in[i].x, h_in[i].y);
    }
    printf("\n");

    testStruct* d_in, * d_results;
    cudaMalloc(& d_in, byte_size);
    cudaMalloc(& d_results, byte_size);

    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((array_size+block_size-1) / (block.x));
    
    test_aos << <grid, block >> > (d_in,d_results,array_size);
    cudaMemcpy(h_res, d_results, byte_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < array_size; i++) {
        printf("x: %d y: %d\n", h_res[i].x, h_res[i].y);
    }
    free(h_in);
    free(h_res);
    cudaFree(d_in);
    cudaFree(d_results);
    //
    */


    //Test SOA
    printf("AOS\n");
    int array_size = LEN;
    int byte_size = sizeof(structArray);
    int block_size = 32;

    structArray* h_in, * h_res;
    h_in = (structArray*)malloc(byte_size);
    h_res = (structArray*)malloc(byte_size);

    for (int i = 0; i < array_size; i++) {
        h_in->x[i] = 1;
        h_in->y[i] = 2;
    }

    printf("OG: \n");
    for (int i = 0; i < array_size; i++) {
        printf("x: %d y: %d\n", h_in->x[i], h_in->y[i]);
    }
    printf("\n");

    structArray* d_in, * d_results;
    cudaMalloc(&d_in, byte_size);
    cudaMalloc(&d_results, byte_size);

    cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((array_size + block_size - 1) / (block.x));

    test_soa << <grid, block >> > (d_in, d_results, array_size);
    cudaMemcpy(h_res, d_results, byte_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < array_size; i++) {
        printf("x: %d y: %d\n", h_res->x[i], h_res->y[i]);
    }
    free(h_in);
    free(h_res);
    cudaFree(d_in);
    cudaFree(d_results);
    //
    return 0;
}