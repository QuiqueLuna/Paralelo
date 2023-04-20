
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>

using namespace std;

#define N 64

struct patients {
    int PatientID[N];
    int age[N]; //12-60
    double glucosa[N]; //100-300
    double heart_rate[N]; //90-170
    double pressure_s[N]; //100-150
    double pressure_d[N]; //70-90
};


__global__ void unroll(int* a, int* b, int n) {
    int tid = threadIdx.x;
    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
    int index = BLOCK_OFFSET + tid;

    int* i_data = a + BLOCK_OFFSET;

    if ((index + blockDim.x) < n) {
        a[index] += a[index + blockDim.x];
    }

    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            i_data += i_data[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        b[blockIdx.x] = i_data[0];
    }
}

__global__ void unrolling_complete(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int* i_data = int_array + blockDim.x * blockIdx.x;

    if (blockDim.x == 1024 && tid < 512)
        i_data[tid] += i_data[tid + 512];
    __syncthreads();

    if (blockDim.x == 512 && tid < 256)
        i_data[tid] += i_data[tid + 256];
    __syncthreads();

    if (blockDim.x == 256 && tid < 128)
        i_data[tid] += i_data[tid + 128];
    __syncthreads();

    if (blockDim.x == 128 && tid < 64)
        i_data[tid] += i_data[tid + 64];
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
    }

}

__global__ void mediaUnrollComplete(patients* data, patients* temp) {
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int* i_data = data->age + blockDim.x * blockIdx.x;
    double* i_data2 = data->glucosa + blockDim.x * blockIdx.x;
    double* i_data3 = data->heart_rate + blockDim.x * blockIdx.x;
    double* i_data4 = data->pressure_d + blockDim.x * blockIdx.x;
    double* i_data5 = data->pressure_s + blockDim.x * blockIdx.x;

    if (blockDim.x == 1024 && tid < 512) {
        i_data[tid] += i_data[tid + 512];
        i_data2[tid] += i_data2[tid + 512];
        i_data3[tid] += i_data3[tid + 512];
        i_data4[tid] += i_data4[tid + 512];
        i_data5[tid] += i_data5[tid + 512];

    }
    __syncthreads();

    if (blockDim.x == 512 && tid < 256) {
        i_data[tid] += i_data[tid + 256];
        i_data2[tid] += i_data2[tid + 512];
        i_data3[tid] += i_data3[tid + 512];
        i_data4[tid] += i_data4[tid + 512];
        i_data5[tid] += i_data5[tid + 512];

    }
    __syncthreads();

    if (blockDim.x == 256 && tid < 128) {
        i_data[tid] += i_data[tid + 128];
        i_data2[tid] += i_data2[tid + 512];
        i_data3[tid] += i_data3[tid + 512];
        i_data4[tid] += i_data4[tid + 512];
        i_data5[tid] += i_data5[tid + 512];

    }
    __syncthreads();

    if (blockDim.x == 128 && tid < 64) {
        i_data[tid] += i_data[tid + 64];
        i_data2[tid] += i_data2[tid + 512];
        i_data3[tid] += i_data3[tid + 512];
        i_data4[tid] += i_data4[tid + 512];
        i_data5[tid] += i_data5[tid + 512];

    }
    __syncthreads();

    if (tid < 32) {
        volatile int* vsmem = i_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
        
        volatile double* vsmem2 = i_data2;
        vsmem2[tid] += vsmem2[tid + 32];
        vsmem2[tid] += vsmem2[tid + 16];
        vsmem2[tid] += vsmem2[tid + 8];
        vsmem2[tid] += vsmem2[tid + 4];
        vsmem2[tid] += vsmem2[tid + 2];
        vsmem2[tid] += vsmem2[tid + 1];
        
        volatile double* vsmem3 = i_data3;
        vsmem3[tid] += vsmem3[tid + 32];
        vsmem3[tid] += vsmem3[tid + 16];
        vsmem3[tid] += vsmem3[tid + 8];
        vsmem3[tid] += vsmem3[tid + 4];
        vsmem3[tid] += vsmem3[tid + 2];
        vsmem3[tid] += vsmem3[tid + 1];
        
        volatile double* vsmem4 = i_data4;
        vsmem4[tid] += vsmem4[tid + 32];
        vsmem4[tid] += vsmem4[tid + 16];
        vsmem4[tid] += vsmem4[tid + 8];
        vsmem4[tid] += vsmem4[tid + 4];
        vsmem4[tid] += vsmem4[tid + 2];
        vsmem4[tid] += vsmem4[tid + 1];
        
        volatile double* vsmem5 = i_data5;
        vsmem5[tid] += vsmem5[tid + 32];
        vsmem5[tid] += vsmem5[tid + 16];
        vsmem5[tid] += vsmem5[tid + 8];
        vsmem5[tid] += vsmem5[tid + 4];
        vsmem5[tid] += vsmem5[tid + 2];
        vsmem5[tid] += vsmem5[tid + 1];
    }

    if (tid == 0) {
        temp->age[blockIdx.x] = i_data[0];
        temp->glucosa[blockIdx.x] = i_data2[0];
    }
    
}

void mediaCPU(patients* data, double* res) {
    for (int i = 0; i < N; i++) {
        res[0] += data->age[i];
        res[1] += data->glucosa[i];
        res[2] += data->heart_rate[i];
        res[3] += data->pressure_s[i];
        res[4] += data->pressure_d[i];
    }
    printf("SUMAS: %f %f %f %f %f\n", res[0],res[1],res[2],res[3], res[4]);
    for (int i = 0; i < 5; i++) {
        res[i] /= N;
    }
}

int main() {
    /*
    patients* host_a, *host_b;
    host_a = (patients*)malloc(sizeof(patients));
    host_b = (patients*)malloc(sizeof(patients));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> d1(12, 60);

    uniform_int_distribution<> d2(100, 300);
    uniform_int_distribution<> d3(90, 170);
    uniform_int_distribution<> d4(100, 150);
    uniform_int_distribution<> d5(70, 90);

    int* aux;
    aux = (int*)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        host_a->PatientID[i] = i+1;
        host_a->age[i] = d1(gen);
        aux[i] = d1(gen);
        host_a->glucosa[i] = d2(gen);
        host_a->heart_rate[i] = d3(gen);
        host_a->pressure_s[i] = d4(gen);
        host_a->pressure_d[i] = d5(gen);
        printf("ID: %d Age: %d Glucosa: %f HeartRate: %f Pressure_d: %f Pressure_d: %f\n", host_a->PatientID[i], host_a->age[i], host_a->glucosa[i], host_a->heart_rate[i], host_a->pressure_s[i], host_a->pressure_d[i]);
    }

    printf("AUX: \n");
    for (int i = 0; i < N; i++) {
        printf("%d ", aux[i]);
    }
    printf("\n");

    int* auxdev;
    patients* d_a, *d_b;
    cudaMalloc(&d_a, sizeof(patients));
    cudaMalloc(&auxdev, sizeof(int)*N);
    cudaMalloc(&d_b, sizeof(patients));
    cudaMemcpy(d_a, host_a, sizeof(patients), cudaMemcpyHostToDevice);
    cudaMemcpy(auxdev, aux, sizeof(patients), cudaMemcpyHostToDevice);

    int blockSize = 32;
    dim3 block(blockSize);
    dim3 grid((N + blockSize - 1) / (block.x));

    int* temp, *tempH;
    tempH = (int*)malloc(sizeof(int)*N);
    cudaMalloc(&temp, sizeof(int)*N);

    unrolling_complete << <8, 128 >> > (auxdev, temp,N);
    cudaMemcpy(aux, auxdev, sizeof(patients), cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_b, d_b, sizeof(patients), cudaMemcpyDeviceToHost);
    cudaMemcpy(tempH, temp, sizeof(patients), cudaMemcpyDeviceToHost);

    printf("A: \n");
    for (int i = 0; i < N; i++) {
        printf("%d ", aux[i]);
    }
    printf("\n");

    printf("B: \n");
    for (int i = 0; i < N; i++) {
        cout << tempH[i] << " ";
    }
    printf("\n");

    int res = 0;
    printf("RES: \n");
    for (int i = 0; i < grid.x; i++) {
        res += tempH[i];
    }
    printf("%d\n", res);
    */


    ////////
    patients* host_a, * host_b;
    host_a = (patients*)malloc(sizeof(patients));
    host_b = (patients*)malloc(sizeof(patients));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> d1(12, 60);

    uniform_int_distribution<> d2(100, 300);
    uniform_int_distribution<> d3(90, 170);
    uniform_int_distribution<> d4(100, 150);
    uniform_int_distribution<> d5(70, 90);

    int suma = 0;
    for (int i = 0; i < N; i++) {
        host_a->PatientID[i] = i + 1;
        host_a->age[i] = d1(gen);
        suma += host_a->age[i];
        host_a->glucosa[i] = d2(gen);
        host_a->heart_rate[i] = d3(gen);
        host_a->pressure_s[i] = d4(gen);
        host_a->pressure_d[i] = d5(gen);
        printf("ID: %d Age: %d Glucosa: %f HeartRate: %f Pressure_d: %f Pressure_d: %f\n", host_a->PatientID[i], host_a->age[i], host_a->glucosa[i], host_a->heart_rate[i], host_a->pressure_s[i], host_a->pressure_d[i]);
    }
    printf("\n");


    //CPU
    double* resCPU;
    resCPU = (double*)malloc(sizeof(double) * 4);
    for (int i = 0; i < 5; i++) {
        resCPU[i] = 0;
    }
    mediaCPU(host_a, resCPU);
    printf("Medias:\nAge: %f Glucosa: %f HeartRate: %f Pressure_d: %f Pressure_d: %f \n", resCPU[0], resCPU[1], resCPU[2], resCPU[3],resCPU[4]);
    printf("\n");


    patients* d_a, * d_b;
    cudaMalloc(&d_a, sizeof(patients));
    cudaMalloc(&d_b, sizeof(patients));
    cudaMemcpy(d_a, host_a, sizeof(patients), cudaMemcpyHostToDevice);


    patients* prueba;
    prueba = (patients*)malloc(sizeof(patients));


    dim3 block(64);
    dim3 grid(1);
    mediaUnrollComplete << <grid, block >> > (d_a, d_b);
    cudaMemcpy(host_a, d_a, sizeof(patients), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, d_b, sizeof(patients), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1; i++) {
        printf("ID: %d Age: %d Glucosa: %f HeartRate: %f Pressure_d: %f Pressure_d: %f\n", host_a->PatientID[i], host_a->age[i], host_a->glucosa[i], host_a->heart_rate[i], host_a->pressure_s[i], host_a->pressure_d[i]);
    }

    double res = host_a->age[0]/(double)N, res2= host_a->glucosa[0] / N, res3= host_a->heart_rate[0]/N, res4= host_a->pressure_s[0]/N, res5= host_a->pressure_d[0] / N;
    printf("Medias:\nAge: %f Glucosa: %f HeartRate: %f Pressure_d: %f Pressure_d: %f \n", res,res2,res3,res4,res5);



    cudaDeviceSynchronize();
    cudaDeviceReset();

    free(host_a);
    free(host_b);

    return 0;
}
