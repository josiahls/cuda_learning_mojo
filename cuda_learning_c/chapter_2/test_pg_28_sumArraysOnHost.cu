#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

// From pg 40
#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("cuda:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}


void initialData(float *ip, int size) {
    // Generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main() {
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    // Initialize the host values and assign the starting data to them.
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // Initialize the device / gpu values
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    // Copy the host data to the device allocated memory
    // Important note:
    //  - dst <- src. The kind tells us which devices / hosts we are moving to.
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // sumArraysOnHost(h_A, h_B, h_C, nElem);
    sumArraysOnGPU<<<1, nElem>>>(d_A, d_B, d_C, nElem);

    // Note cudaMemCpy is always device, host
    CHECK(cudaMemcpy(h_C, d_C,  nBytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < nElem; i++) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}