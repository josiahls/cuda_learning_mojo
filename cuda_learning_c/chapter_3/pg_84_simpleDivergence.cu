//===---------------------------BOILER PLATE--------------------------------====
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

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

float *host_malloc(int nBytes) { return (float *)malloc(nBytes); }

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(int *ip, int size) {
  for (int i = 0; i < size; i++) {
    ip[i] = i;
  }
}

void initialData(float *fp, int size) {
  // Generate different seed for random number
  time_t t;
  srand((unsigned int)time(&t));
  for (int i = 0; i < size; i++) {
    fp[i] = (float)(rand() & 0xFF) / 10.0f;
  }
}

void checkResult(float *h_C, float *h_GpuCheck, int nElem) {
  bool match = true;
  for (int i = 0; i < nElem; i++) {
    if (abs(h_C[i] - h_GpuCheck[i]) > 1e-5) {
      printf("Error at index %d: %f != %f\n", i, h_C[i], h_GpuCheck[i]);
      match = false;
      break;
    }
  }
  if (match) {
    printf("Test passed\n");
  } else {
    printf("Test failed\n");
  }
}

inline void printElaps(double &iStart,double &iElaps, const char * function_name, const dim3 &grid, const dim3 &block) {
  iElaps = cpuSecond() - iStart;
  printf("%s <<< %4d, %4d >>> elapsed %f sec \n", function_name, grid.x, block.x, iElaps);
}

//===---------------------------BOILER PLATE--------------------------------====

__global__ void mathKernel1(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a,b;
  a = b = 0.0f;

  if (tid % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void warmingUp(float *c) { 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a,b;
  a = b = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void mathKernel2(float *c) { 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a,b;
  a = b = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}


__global__ void mathKernel3(float *c) { 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia,ib;
  ia = ib = 0.0f;

  bool ipred = (tid % 2 == 0);
  if (ipred) {
    ia = 100.0f;
  } 
  if (!ipred) {
    ib = 200.0f;
  }
  c[tid] = ia + ib;
}



int main(int argc, char **argv) {
  const int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("%s Using device %d: %s \n", argv[0], dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  int size = 64;
  int blockSize = 64;
  if (argc > 1) blockSize = atoi(argv[1]);
  if (argc > 2) size = atoi(argv[2]);
  printf("Data size %d ", size);

  dim3 block (blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);

  printf("Execution Configure (block %d, grid %d)\n", block.x, grid.x);

  float *d_C;
  size_t NBytes = size * sizeof(float);

  cudaMalloc((float**)&d_C, NBytes);

  double iStart, iElips;
  cudaDeviceSynchronize();
  iStart = cpuSecond();
  warmingUp<<< grid, block>>> (d_C);
  cudaDeviceSynchronize();
  printElaps(iStart,iElips, "warmpingUp", grid, block);

  iStart = cpuSecond();
  mathKernel1<<< grid, block>>> (d_C);
  cudaDeviceSynchronize();
  printElaps(iStart,iElips, "mathKernel1", grid, block);

  iStart = cpuSecond();
  mathKernel2<<< grid, block>>> (d_C);
  cudaDeviceSynchronize();
  printElaps(iStart,iElips, "mathKernel2", grid, block);

  iStart = cpuSecond();
  mathKernel3<<< grid, block>>> (d_C);
  cudaDeviceSynchronize();
  printElaps(iStart,iElips, "mathKernel3", grid, block);

  cudaFree(d_C);
  cudaDeviceReset();



  return 0;
}