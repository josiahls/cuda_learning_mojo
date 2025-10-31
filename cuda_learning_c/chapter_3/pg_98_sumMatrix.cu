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

void host_malloc(float **ptr, int nBytes) {  
  *ptr = (float *)malloc(nBytes); 
}

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

void hostSumMatrix2D(float *A, float *B, float *C, int NX, int NY) {
  for (int idx = 0; idx < NX * NY; idx++) {
    C[idx] = A[idx] + B[idx];
  }
}

__global__ void sumMatrix2D(float *A, float *B, float *C, int NX, int NY) {
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = iy * NX + ix;

  if (ix < NX && iy < NY) {
    C[idx] = A[idx] + B[idx];
  }
}


int main(int argc, char **argv) {

  /**
  Evaluated via:

  pixi run cudad cuda_learning_c/chapter_3 pg_84_simpleDivergence
  ncu -f --metrics smsp__sass_branch_targets_threads_divergent.avg,smsp__sass_branch_targets_threads_divergent.sum build/cuda_learning_c/chapter_3/pg_84_simpleDivergence 
  
  */
  const int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("%s Using device %d: %s \n", argv[0], dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  int nx = 1<<14;
  int ny = 1<<14;
  int nxy = nx * ny;
  size_t NBytes = nxy * sizeof(float);

  int dimx = 16;
  int dimy = 16;

  if (argc > 1) dimx = atoi(argv[1]);
  if (argc > 2) dimy = atoi(argv[2]);
  printf("Data size x: %d y: %d\n", dimx, dimy);

  dim3 block (dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y) / block.y);

  printf("Execution Configure (block %d, grid %d)\n", block.x, grid.x);

  float *h_A, *h_B, *h_C, *h_check_C;

  host_malloc((float**)&h_A, NBytes);
  host_malloc((float**)&h_B, NBytes);
  host_malloc((float**)&h_C, NBytes);
  host_malloc((float**)&h_check_C, NBytes);

  initialData(h_A, nxy);
  initialData(h_B, nxy);

  float *d_A, *d_B, *d_C;

  cudaMalloc((float**)&d_A, NBytes);
  cudaMalloc((float**)&d_B, NBytes);
  cudaMalloc((float**)&d_C, NBytes);

  cudaMemcpy(d_A, h_A, NBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, NBytes, cudaMemcpyHostToDevice);

  hostSumMatrix2D(h_A, h_B, h_check_C, nx,ny);

  double iStart, iElips;
  iStart = cpuSecond();
  sumMatrix2D<<< grid, block>>> (d_A, d_B, d_C, nx, ny);
  cudaDeviceSynchronize();
  printElaps(iStart,iElips, "sumMatrix2D", grid, block);

  cudaMemcpy(h_C, d_C, NBytes, cudaMemcpyDeviceToHost);

  checkResult(h_check_C, h_C, nxy);

  cudaFree(d_C);
  cudaDeviceReset();

  return 0;
}