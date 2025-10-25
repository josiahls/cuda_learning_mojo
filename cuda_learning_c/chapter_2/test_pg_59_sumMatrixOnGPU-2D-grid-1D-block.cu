//===---------------------------BOILER PLATE--------------------------------====
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
//===---------------------------BOILER PLATE--------------------------------====

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx;
    ib += nx;
    ic += nx;
  }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny) {

  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  unsigned int idx = iy * nx + ix;

  if (ix < nx && iy < ny) {
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);

  const int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using device %d: %s \n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  const int nx = 1 << 14; // 14 16,384 elements.
  const int ny = 1 << 14;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // malloc host memory
  float *h_A, *h_B, *h_Ref, *h_GpuRef;
  h_A = host_malloc(nBytes);
  h_B = host_malloc(nBytes);
  h_Ref = host_malloc(nBytes);
  h_GpuRef = host_malloc(nBytes);

  double iStart = cpuSecond();
  initialData(h_A, nxy);
  initialData(h_B, nxy);
  double iElaps = cpuSecond() - iStart;
  printf("initialData Time: %3f\n", iElaps);

  memset(h_Ref, 0, nBytes);
  memset(h_GpuRef, 0, nBytes);

  iStart = cpuSecond();
  sumMatrixOnHost(h_A, h_B, h_Ref, nx, ny);
  iElaps = cpuSecond() - iStart;
  printf("sumMatrixOnHost Time: %3f\n", iElaps);

  // amlloc device global memory
  float *d_MatA, *d_MatB, *d_MatC;
  cudaMalloc((void **)&d_MatA, nBytes);
  cudaMalloc((void **)&d_MatB, nBytes);
  cudaMalloc((void **)&d_MatC, nBytes);

  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  const int dimx = 256;

  dim3 block(dimx, 1);
  dim3 grid((nx + block.x - 1) / block.x, ny);

  iStart = cpuSecond();
  sumMatrixOnGPUMix <<< grid, block >>> (d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;
  printf("sumMatrixOnGPUMix <<< (%d, %d), (%d, %d) >>> Time: %3f\n", grid.x,
         grid.y, block.x, block.y, iElaps);

  cudaMemcpy(h_GpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

  checkResult(h_Ref, h_GpuRef, nxy);

  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  free(h_A);
  free(h_B);
  free(h_Ref);
  free(h_GpuRef);

  cudaDeviceReset();
  /**
  For a GTX 1660 Ti:

  Block Size: 32, 1
    sumMatrixOnHost Time: 0.854107
    sumMatrixOnGPU1D <<< (512, 1), (32, 1) >>> Time: 0.020902

  */
  return 0;
}