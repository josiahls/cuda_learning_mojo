#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello, World from GPU! thread: %d\n", threadIdx.x);
}

__global__ void helloFromGPUOnlyThread5() {
    if (threadIdx.x == 5) {
        printf("Hello, World from GPU! thread: %d\n", threadIdx.x);
    }
}

int main() {
    printf("Hello, World from CPU!\n");
    // helloFromGPU<<<1, 6>>>();
    helloFromGPUOnlyThread5<<<1, 6>>>();
    // When this is removed, helloFromGPU never gets executed in the first place!
    // cudaDeviceSynchronize();
    cudaDeviceReset();

    // cudaDeviceSynchonize(); can replace the cudaDeviceReset();
    return 0;
}