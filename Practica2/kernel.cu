#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print() {
    printf("threadIdx %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);

}

int main(){
    int nx=4;
    int ny=4;
    int nz=4;
    dim3 block(2, 2, 2);
    dim3 grid(nx / block.x, ny / block.y, nz/block.z);
    print << <grid, block >> > ();

    return 0;
}