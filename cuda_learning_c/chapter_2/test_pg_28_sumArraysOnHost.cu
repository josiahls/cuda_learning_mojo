#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

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
    // NOTE: Doing this will only work if there is only 1 block!
    // int i = threadIdx.x;
    // NOTE: This scales to multiple blocks!
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    int nElem = 1 << 24;
    size_t nBytes = nElem * sizeof(float);

    // Initialize the host values and assign the starting data to them.
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);
    float *h_C_check = (float *)malloc(nBytes);
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

    // dim3 block(nElem / 2);
    // dim3 grid(nElem / block.x);
    printf("Vector size: %d\n", nElem);
    dim3 block(1023);
    dim3 grid((nElem + block.x - 1) / block.x);

    // Sum the host data to check the result
    double iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, h_C_check, nElem);
    double iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost: %f sec\n", iElaps);

    iStart = cpuSecond();
    // sumArraysOnHost(h_A, h_B, h_C, nElem);
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnGPU <<<%d, %d>>>: %f sec\n", grid.x, block.x, iElaps);



    // Note cudaMemCpy is always device, host
    CHECK(cudaMemcpy(h_C, d_C,  nBytes, cudaMemcpyDeviceToHost));

    bool match = true;
    for (int i = 0; i < nElem; i++) {
        if (abs(h_C[i] - h_C_check[i]) > 1e-5) {
            printf("Error at index %d: %f != %f\n", i, h_C[i], h_C_check[i]);
            match = false;
            break;
        }
    }
    if (match) {
        printf("Test passed\n");
    } else {
        printf("Test failed\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}