#include <stdio.h>

__global__ void hello_world() {
    printf("Hello, World from GPU!\n");
}

int main() {
    printf("Hello, World from CPU!\n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}