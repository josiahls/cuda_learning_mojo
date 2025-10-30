#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // NOTE: What was the point of this exercise? This appears to just demo basic
    // struct fields + math. 
    int nElem = 1024;

    dim3 block = (nElem);
    dim3 grid = ((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = (512);
    grid.x = ((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = (256);
    grid.x = ((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = (128);
    grid.x = ((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    cudaDeviceReset();
    return 0;
}