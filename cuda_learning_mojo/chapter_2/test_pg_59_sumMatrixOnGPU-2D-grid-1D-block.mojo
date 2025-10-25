import sys
import time
from random import rand, seed
from gpu import thread_idx,block_idx,block_dim
from gpu.host import DeviceContext, Dim
from gpu.host.info import GPUInfo, _accelerator_arch
from memory import memset
from gpu.random import Random


alias Float32Ptr = UnsafePointer[Float32]

struct Timer:
  var fn_name: String
  var start_sec:Float64

  fn __init__(out self, var fn_name:String):
    self.fn_name = fn_name
    self.start_sec = time.perf_counter_ns() / 1e9

  fn __enter__(ref self) -> ref[self] Self: 
    return self

  fn __exit__(self):
    var end_sec = time.perf_counter_ns() / 1e9
    print(self.fn_name, 'elapsed time: ', end_sec - self.start_sec, 'seconds')

fn initialize_data[n_elem: UInt](ptr: Float32Ptr):
    var rng = Random(seed=time.perf_counter_ns())
    alias left_over = n_elem % 4
    var tracker: Int = 0

    for i in range(0, n_elem - left_over, 4):
      var uniform_values = rng.step_uniform()
      @parameter
      for k in range(4):
        ptr[i + UInt(k)] = uniform_values[k]
        tracker = i + UInt(k)
    
    @parameter
    if left_over > 0:
      var uniform_values = rng.step_uniform()
      @parameter
      for k in range(left_over):
        ptr[tracker  + k] = uniform_values[k]



fn check_result(h_C:Float32Ptr, h_GpuCheck:Float32Ptr, n_elem: Int):
  for i in range(n_elem):
    if (abs(h_C[i] - h_GpuCheck[i]) > 1e-5):
      print("Error at index", i,":", h_C[i],"!=", h_GpuCheck[i])
      print('Test Failed')
      return
   
  print("Test passed")


fn sum_matrix_on_host[
  nx:Int,
  ny:Int
](A:Float32Ptr, B:Float32Ptr, C:Float32Ptr):
  var ia:Float32Ptr = A
  var ib:Float32Ptr = B
  var ic:Float32Ptr = C

  for iy in range(ny):
    for ix in range(nx):
      ic[ix] = ia[ix] + ib[ix]

    ia += nx
    ib += nx
    ic += nx


fn sum_matrix_on_gpu_mix[
  nx:UInt,
  ny:UInt
](A:Float32Ptr, B:Float32Ptr, C:Float32Ptr):
  var ix:UInt = thread_idx.x + block_idx.x * block_dim.x
  var iy:UInt = block_idx.y
  var idx:UInt = iy * nx + ix


  if (ix < nx and iy < ny):
    C[idx] = A[idx] + B[idx]


fn main() raises:
    print(sys.argv()[0], "Starting...")

    alias dev:Int = 0
    gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()
    print("Using Device ", gpu_info)

    alias nx: Int = 1 << 14 # 14 16,384 elements.
    alias ny: Int = 1 << 14

    alias nxy: UInt = UInt(nx * ny)
    alias nBytes: Int = nxy * UInt(sys.info.size_of[Float32]())
    print("Matrix size: nx", nx, "ny", ny)

    # malloc host memory
    var h_A = Float32Ptr.alloc(nxy)
    var h_B = Float32Ptr.alloc(nxy)
    var h_Ref = Float32Ptr.alloc(nxy)
    var h_GpuRef = Float32Ptr.alloc(nxy)

    with Timer('initialize_data'):
      initialize_data[nxy](h_A)
      initialize_data[nxy](h_B)

    memset(h_Ref, 0, nxy)
    memset(h_GpuRef, 0, nxy)

    with Timer('sum_matrix_on_host'):
      sum_matrix_on_host[nx, ny](h_A, h_B, h_Ref)

    with DeviceContext(dev) as ctx:
      var d_MatA = ctx.enqueue_create_buffer[Float32.dtype](nxy)
      var d_MatB = ctx.enqueue_create_buffer[Float32.dtype](nxy)
      var d_MatC = ctx.enqueue_create_buffer[Float32.dtype](nxy)

      ctx.enqueue_copy(d_MatA, h_A)
      ctx.enqueue_copy(d_MatB, h_B)

      alias dimx: Int = 32
      alias dimy: Int = 1

      var block: Dim = Dim(dimx, dimy)
      var grid = Dim(
        Int((nx + block.x() - 1) / block.x()), ny #, 
        # Int((ny + block.y() - 1) / block.y())
      )

      print('Block:', block, "grid:", grid)

      with Timer("sum_matrix_on_gpu_mix <<< ({}, {}), ({}, {}) >>>".format(
        grid.x(),grid.y(), block.x(), block.y()
      )):
        ctx.enqueue_function[sum_matrix_on_gpu_mix[UInt(nx),UInt(ny)]](
          d_MatA, d_MatB, d_MatC,
          grid_dim=grid,
          block_dim=block
        )
        ctx.synchronize()

      ctx.enqueue_copy(h_GpuRef, d_MatC)

    check_result(h_Ref, h_GpuRef, nxy)

