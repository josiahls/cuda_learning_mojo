//===---------------------------BOILER PLATE--------------------------------====
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));       \
      /*  The negative exit value indicates a cuda error */                    \
      exit(-10 * error);                                                       \
    }                                                                          \
  }



__global__ void nestedHelloWorld(int const iSize, int iDepth) {
    int tid = threadIdx.x;
    printf("Recursion=%d :Hello World from thread %d block %d\n",
        iDepth, tid,blockIdx.x);

    if (iSize == 1) return;

    // Decrease the number of threads by the power of 2 (rshift)
    int nthreads = iSize >> 1;

    if (tid == 0 && nthreads > 0 ) {
        // Dynamic parallelism - requires -rdc=true for nvcc compilation
        // clangd doesn't support CUDA dynamic parallelism checking
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth); // clangd-ignore
        printf("--------> nested execution depth: %d\n",iDepth);
    }

}


int main(int argc, char **argv) {
    const int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s Using device %d: %s \n", argv[0], dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1<<4;
    int dimx = 512;
    if (argc > 1) dimx = atoi(argv[1]);
    printf("Data size x: %d y: %d\n", dimx, 1);
  
    dim3 block (dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    nestedHelloWorld<<<grid, block>>>(nx, 0);

    CHECK(cudaDeviceSynchronize());
}