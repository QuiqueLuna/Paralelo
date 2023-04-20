
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <random>
#include <cmath>

using namespace std;

#define N 1024

struct patients {
    int PatientID[N];
    int age[N]; //12-60
    double glucosa[N]; //100-300
    double heart_rate[N]; //90-170
    double pressure_s[N]; //100-150
    double pressure_d[N]; //70-90
};

__global__ void mediaDesviacionUnrollComplete(patients* data, patients* temp) {
    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int* i_data = data->age + blockDim.x * blockIdx.x;
    double* i_data2 = data->glucosa + blockDim.x * blockIdx.x;
    double* i_data3 = data->heart_rate + blockDim.x * blockIdx.x;
    double* i_data4 = data->pressure_s + blockDim.x * blockIdx.x;
    double* i_data5 = data->pressure_d + blockDim.x * blockIdx.x;

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
        i_data2[tid] += i_data2[tid + 256];
        i_data3[tid] += i_data3[tid + 256];
        i_data4[tid] += i_data4[tid + 256];
        i_data5[tid] += i_data5[tid + 256];

    }
    __syncthreads();

    if (blockDim.x == 256 && tid < 128) {
        i_data[tid] += i_data[tid + 128];
        i_data2[tid] += i_data2[tid + 128];
        i_data3[tid] += i_data3[tid + 128];
        i_data4[tid] += i_data4[tid + 128];
        i_data5[tid] += i_data5[tid + 128];

    }
    __syncthreads();

    if (blockDim.x == 128 && tid < 64) {
        i_data[tid] += i_data[tid + 64];
        i_data2[tid] += i_data2[tid + 64];
        i_data3[tid] += i_data3[tid + 64];
        i_data4[tid] += i_data4[tid + 64];
        i_data5[tid] += i_data5[tid + 64];

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
        temp->heart_rate[blockIdx.x] = i_data3[0];
        temp->pressure_s[blockIdx.x] = i_data4[0];
        temp->pressure_d[blockIdx.x] = i_data5[0];
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
    //printf("SUMAS: %f %f %f %f %f\n", res[0], res[1], res[2], res[3], res[4]);
    for (int i = 0; i < 5; i++) {
        res[i] /= N;
    }
}

void desviacionCPU(patients* data, double* res, double m1, double m2, double m3, double m4, double m5) {
    for (int i = 0; i < N; i++) {
        res[0] += (data->age[i]-(int)m1)* (data->age[i] - (int)m1);
        res[1] += (data->glucosa[i]-m2)* (data->glucosa[i] - m2);
        res[2] += (data->heart_rate[i]-m3)* (data->heart_rate[i] - m3);
        res[3] += (data->pressure_s[i]-m4)* (data->pressure_s[i] - m4);
        res[4] += (data->pressure_d[i]-m5)* (data->pressure_d[i] - m5);
    }
    //printf("SUMAS: %f %f %f %f %f\n", res[0], res[1], res[2], res[3], res[4]);
    for (int i = 0; i < 5; i++) {
        res[i] = sqrt(res[i]/(N-2));
    }
}

int main() {
    patients* host_a, * host_b, *hostAUX, *hostAUX2;
    host_a = (patients*)malloc(sizeof(patients));
    host_b = (patients*)malloc(sizeof(patients));
    hostAUX = (patients*)malloc(sizeof(patients));
    hostAUX2 = (patients*)malloc(sizeof(patients));

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> d1(12, 60);

    uniform_int_distribution<> d2(100, 300);
    uniform_int_distribution<> d3(90, 170);
    uniform_int_distribution<> d4(100, 150);
    uniform_int_distribution<> d5(70, 90);

    for (int i = 0; i < N; i++) {
        host_a->PatientID[i] = i + 1;
        host_a->age[i] = d1(gen);
        host_a->glucosa[i] = d2(gen);
        host_a->heart_rate[i] = d3(gen);
        host_a->pressure_s[i] = d4(gen);
        host_a->pressure_d[i] = d5(gen);
        hostAUX->PatientID[i] = host_a->PatientID[i];
        hostAUX->age[i] = host_a->age[i];
        hostAUX->glucosa[i] = host_a->glucosa[i];
        hostAUX->heart_rate[i] = host_a->heart_rate[i];
        hostAUX->pressure_s[i] = host_a->pressure_s[i];
        hostAUX->pressure_d[i] = host_a->pressure_d[i];
        //printf("ID: %d Age: %d Glucosa: %f HeartRate: %f Pressure_s: %f Pressure_d: %f\n", host_a->PatientID[i], host_a->age[i], host_a->glucosa[i], host_a->heart_rate[i], host_a->pressure_s[i], host_a->pressure_d[i]);
    }
    printf("\n");

    //CPU Media
    double* resCPU;
    resCPU = (double*)malloc(sizeof(double) * 5);
    for (int i = 0; i < 5; i++) {
        resCPU[i] = 0;
    }
    printf("CPU Medias\n");
    mediaCPU(host_a, resCPU);
    printf("Age: %f Glucosa: %f HeartRate: %f Pressure_s: %f Pressure_d: %f \n", resCPU[0], resCPU[1], resCPU[2], resCPU[3], resCPU[4]);
    printf("\n");

    //CPU Desviacion
    double* dresCPU;
    dresCPU = (double*)malloc(sizeof(double) * 5);
    for (int i = 0; i < 5; i++) {
        dresCPU[i] = 0;
    }
    printf("CPU Desviacion\n");
    desviacionCPU(host_a, dresCPU, resCPU[0], resCPU[1], resCPU[2], resCPU[3], resCPU[4]);
    printf("Age: %f Glucosa: %f HeartRate: %f Pressure_s: %f Pressure_d: %f \n", dresCPU[0], dresCPU[1], dresCPU[2], dresCPU[3], dresCPU[4]);
    printf("\n");

    //GPU Media
    patients* d_a, * d_b, *d_a2, *d_b2;
    cudaMalloc(&d_a, sizeof(patients));
    cudaMalloc(&d_b, sizeof(patients));
    cudaMalloc(&d_a2, sizeof(patients));
    cudaMalloc(&d_b2, sizeof(patients));
    cudaMemcpy(d_a, host_a, sizeof(patients), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid(8);
    mediaDesviacionUnrollComplete << <grid, block >> > (d_a, d_b);
    cudaMemcpy(host_a, d_a, sizeof(patients), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_b, d_b, sizeof(patients), cudaMemcpyDeviceToHost);

    double mediaSum[5] = {0};
    for (int i = 0; i < grid.x; i++) {
        mediaSum[0]+=host_b->age[i];
        mediaSum[1]+=host_b->glucosa[i];
        mediaSum[2]+=host_b->heart_rate[i];
        mediaSum[3]+=host_b->pressure_s[i];
        mediaSum[4]+=host_b->pressure_d[i];
    }
    //printf("SUMAS GPU: %f %f %f %f %f\n", mediaSum[0], mediaSum[1], mediaSum[2], mediaSum[3], mediaSum[4]);
    

    printf("GPU Medias\n");
    double res = mediaSum[0] / (double)N, res2 = mediaSum[1] / N, res3 = mediaSum[2] / N, res4 = mediaSum[3]  / N, res5 = mediaSum[4] / N;
    printf("Age: % f Glucosa: % f HeartRate: % f Pressure_s: % f Pressure_d: % f \n", res, res2, res3, res4, res5);
    printf("\n");
    

    //GPU Desviacion
    for (int i = 0; i < N; i++) {
        hostAUX->age[i]-=(int)res;
        hostAUX->glucosa[i]-=res2;
        hostAUX->heart_rate[i]-=res3;
        hostAUX->pressure_s[i]-=res4;
        hostAUX->pressure_d[i]-=res5;
        hostAUX->age[i] *= hostAUX->age[i];
        hostAUX->glucosa[i] *= hostAUX->glucosa[i];
        hostAUX->heart_rate[i] *= hostAUX->heart_rate[i];
        hostAUX->pressure_s[i] *= hostAUX->pressure_s[i];
        hostAUX->pressure_d[i] *= hostAUX->pressure_d[i];
        //printf("ID: %d Age: %d Glucosa: %f HeartRate: %f Pressure_d: %f Pressure_d: %f\n", hostAUX->PatientID[i], hostAUX->age[i], hostAUX->glucosa[i], hostAUX->heart_rate[i], hostAUX->pressure_s[i], hostAUX->pressure_d[i]);
    }
    cudaMemcpy(d_a2, hostAUX, sizeof(patients), cudaMemcpyHostToDevice);
    mediaDesviacionUnrollComplete << <grid, block >> > (d_a2,d_b2);
    cudaMemcpy(hostAUX, d_a2, sizeof(patients), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostAUX2, d_b2, sizeof(patients), cudaMemcpyDeviceToHost);

    double desviacionSum[5] = { 0 };
    for (int i = 0; i < grid.x; i++) {
        desviacionSum[0] += hostAUX2->age[i];
        desviacionSum[1] += hostAUX2->glucosa[i];
        desviacionSum[2] += hostAUX2->heart_rate[i];
        desviacionSum[3] += hostAUX2->pressure_s[i];
        desviacionSum[4] += hostAUX2->pressure_d[i];
    }
    
    printf("GPU Desviacion\n");
    double dres = sqrt(desviacionSum[0] / (double)(N-2)), dres2 = sqrt(desviacionSum[1] / (N-2)), dres3 = sqrt(desviacionSum[2] / (N-2)), dres4 = sqrt(desviacionSum[3] / (N-2)), dres5 = sqrt(desviacionSum[4] / (N-2));
    printf("Age: %f Glucosa: %f HeartRate: %f Pressure_s: %f Pressure_d: %f \n", dres, dres2, dres3, dres4, dres5);


    cudaDeviceSynchronize();
    cudaDeviceReset();

    free(host_a);
    free(host_b);
    free(hostAUX);
    free(hostAUX2);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_a2);
    cudaFree(d_b2);

    return 0;
}
