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
  iElaps = (cpuSecond() - iStart) * 1000;
  printf("%s <<< %4d, %4d >>> elapsed %f ms \n", function_name, grid.x, block.x, iElaps);
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

  ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active build/cuda_learning_c/chapter_3/pg_98_sumMatrix 32 32
  ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active build/cuda_learning_c/chapter_3/pg_98_sumMatrix 32 16
  ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active build/cuda_learning_c/chapter_3/pg_98_sumMatrix 16 32
  ncu --metric sm__warps_active.avg.pct_of_peak_sustained_active build/cuda_learning_c/chapter_3/pg_98_sumMatrix 16 16

  ncu --metric 	l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second build/cuda_learning_c/chapter_3/pg_98_sumMatrix 32 32
  ncu --metric 	l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second build/cuda_learning_c/chapter_3/pg_98_sumMatrix 32 16
  ncu --metric 	l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second build/cuda_learning_c/chapter_3/pg_98_sumMatrix 16 32
  ncu --metric 	l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second build/cuda_learning_c/chapter_3/pg_98_sumMatrix 16 16


  ncu --metric 	l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum build/cuda_learning_c/chapter_3/pg_98_sumMatrix 32 32
  ncu --metric 	l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum build/cuda_learning_c/chapter_3/pg_98_sumMatrix 32 16
  ncu --metric 	l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum build/cuda_learning_c/chapter_3/pg_98_sumMatrix 16 32
  ncu --metric 	l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum build/cuda_learning_c/chapter_3/pg_98_sumMatrix 16 16

  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 64 2
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 64 4
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 64 8
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 128 2
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 128 4
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 128 8
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 256 2
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 256 4
  build/cuda_learning_c/chapter_3/pg_98_sumMatrix 256 8
  
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
  // 16x16 -> 13.2 ms -O3 -> 13.32 ms
  // 32x32 -> 13.4 ms -O3 -> 13.41 ms
  // 16x32 -> 13.39 ms -O3 -> 13.38 ms
  // 32x16 -> 13.23 ms -O3 -> 13.29 ms



  if (argc > 1) dimx = atoi(argv[1]);
  if (argc > 2) dimy = atoi(argv[2]);
  printf("Data size x: %d y: %d\n", dimx, dimy);

  dim3 block (dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  printf("Execution Configure (block %d %d, grid %d %d)\n", block.x, block.y, grid.x, grid.y);

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
  CHECK(cudaDeviceSynchronize());
  printElaps(iStart,iElips, "sumMatrix2D", grid, block);

  cudaMemcpy(h_C, d_C, NBytes, cudaMemcpyDeviceToHost);

  checkResult(h_check_C, h_C, nxy);

  cudaFree(d_C);
  cudaDeviceReset();

  return 0;
}