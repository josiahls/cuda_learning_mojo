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



  /**
  Handon, where is iDepth in memory? Is it on the host? I guess it
  gets copied into the global memory in the gpu. But then we are doing an
  in place argument modification. I'm guessing then this argument is copied 
  into shared memory.
  */
__global__ void nestedHelloWorld(int const isize, int idepth) {
	int tid = threadIdx.x;
	// printf("This gets called: isize=%d idepth=%d\n", isize, idepth);
	// if (idepth > 0) {
	// }
	if (isize == 2 && tid == 0) {
		printf("Recursion=%d :Hello World from thread %d block %d stride %d\n",
			idepth, tid, blockIdx.x, isize);
		return;
	}

	if (isize < 2) {
		printf("This line should never be called\n");
		return;
	}


	printf("Recursion=%d :Hello World from thread %d block %d stride %d\n",
		idepth, tid,blockIdx.x, isize);

	// Decrease the number of threads by the power of 2 (rshift)
	int nthreads = isize >> 1;
	if (tid == 0 && nthreads > 0 ) {
		// printf("This gets called: nthreads=%d idepth=%d\n", nthreads, idepth);
		// Dynamic parallelism - requires -rdc=true for nvcc compilation
		// clangd doesn't support CUDA dynamic parallelism checking
		nestedHelloWorld<<<1, nthreads, 0, cudaStreamTailLaunch>>>(nthreads, ++idepth); // clangd-ignore
	}
}


__global__ void nestedHelloWorldSolution(int const iSize, int minSize, int iDepth)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d\n", iDepth, tid);

    // condition to stop recursive execution
    if (iSize == minSize) return;

    // reduce nthreads by half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        int blocks = (nthreads + blockDim.x - 1) / blockDim.x;
        nestedHelloWorld<<<blocks, blockDim.x>>>(nthreads, minSize, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}


__global__ void nestedHelloWorldDepthLimited(int const isize, int idepth, int const maxDepth) {
	int tid = threadIdx.x;
	if (isize == 2 && tid == 0) {
		printf("Recursion(depth limited)=%d :Hello World from thread %d block %d stride %d\n",
			idepth, tid, blockIdx.x, isize);
		return;
	}

	if (isize < 2) {
		printf("This line should never be called\n");
		return;
	}


	printf("Recursion(depth limited)=%d :Hello World from thread %d block %d stride %d\n",
		idepth, tid,blockIdx.x, isize);

	// Decrease the number of threads by the power of 2 (rshift)
	int nthreads = isize >> 1;
	if (nthreads > 0 && idepth >= maxDepth) {
		return;
	} else if (tid == 0 && nthreads > 0 && idepth < maxDepth) {
		// printf("This gets called: nthreads=%d idepth=%d\n", nthreads, idepth);
		// Dynamic parallelism - requires -rdc=true for nvcc compilation
		// clangd doesn't support CUDA dynamic parallelism checking
		nestedHelloWorldDepthLimited<<<1, nthreads, 0, cudaStreamTailLaunch>>>(
			nthreads, 
			++idepth, 
		maxDepth); // clangd-ignore
	}
}


__global__ void nestedHelloWorldDepthLimitedSolution(int const iSize, int iDepth, int maxDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
           blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1 || iDepth >= maxDepth) return;

    // reduce block size to half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth, maxDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}



int main(int argc, char **argv) {
    const int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s Using device %d: %s \n", argv[0], dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nthreads = 64;
    int nblocks = 128;
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, nblocks + 1);
    if (argc > 1) nthreads = atoi(argv[1]);
    int nx = nthreads * nblocks;
    printf("Data size x: %d y: %d\n", nthreads, 1);
  
    dim3 block (nthreads, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    nestedHelloWorld<<<grid, block>>>(nx, 0);


    CHECK(cudaDeviceSynchronize());
	nestedHelloWorldDepthLimited<<<grid, block>>>(nx, 0, 2);

    CHECK(cudaDeviceSynchronize());
}