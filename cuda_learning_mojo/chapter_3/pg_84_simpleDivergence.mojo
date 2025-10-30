import sys
import time
from random import rand, seed
from gpu import thread_idx,block_idx,block_dim, WARP_SIZE
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

fn initialize_data[n_elem: Int](ptr: Float32Ptr):
    var rng = Random(seed=time.perf_counter_ns())
    alias left_over = Int(n_elem % 4)
    var tracker: Int = 0

    for i in range(0, n_elem - left_over, 4):
      var uniform_values = rng.step_uniform()
      @parameter
      for k in range(4):
        ptr[i + k] = uniform_values[k]
        tracker = i + k
    
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


fn warming_up(c:Float32Ptr):
  var tid:Int = Int(block_idx.x * block_dim.x + thread_idx.x)
  var a:Float32 = 0.0
  var b:Float32 = 0.0

  if ((tid / WARP_SIZE) % 2 == 0):
    a = 100.0
  else:
    b = 200.0
  
  c[tid] = a + b;


fn math_kernel_1(c:Float32Ptr):
  var tid:Int = Int(block_idx.x * block_dim.x + thread_idx.x)
  var a:Float32 = 0.0
  var b:Float32 = 0.0

  if (tid % 2 == 0):
    a = 100.0
  else:
    b = 200.0

  c[tid] = a + b


fn math_kernel_2(c:Float32Ptr):
  var tid:Int = Int(block_idx.x * block_dim.x + thread_idx.x)
  var a:Float32 = 0.0
  var b:Float32 = 0.0

  if ((tid / WARP_SIZE) % 2 == 0):
    a = 100.0
  else:
    b = 200.0

  c[tid] = a + b;


fn math_kernel_3(c:Float32Ptr):
  var tid:Int = Int(block_idx.x * block_dim.x + thread_idx.x)
  var ia:Float32 = 0.0
  var ib:Float32 = 0.0

  var ipred:Bool = (tid % 2 == 0)
  if (ipred):
    ia = 100.0
  
  if (not ipred):
    ib = 200.0

  c[tid] = ia + ib


def main():
  var dev:Int = 0
  gpu_info: GPUInfo = GPUInfo.from_name[_accelerator_arch()]()
  print("Using Device ", gpu_info)

  var size:Int = 64
  var block_size:Int = 64
  var n_bytes = size * sys.info.size_of[Float32]()
  if (len(sys.argv()) > 1): block_size = Int(sys.argv()[1])
  if (len(sys.argv()) > 2): size = Int(sys.argv()[2])
  print("Data size ", size);

  var block: Dim = (block_size, 1)
  var grid: Dim = {Int((size + block.x() - 1) / block.x()), 1}

  print("Execution Configure (block", block.x(), ", grid", grid.x(), ")")

  with DeviceContext(dev) as ctx:
    d_C = ctx.create_buffer_sync[Float32.dtype](size)
    
    with Timer("warming_up"):
      ctx.enqueue_function[warming_up](d_C, grid_dim=grid, block_dim=block)
      ctx.synchronize()
    
    with Timer("math_kernel_1"):
      # Should have branch divergence
      ctx.enqueue_function[math_kernel_1](d_C, grid_dim=grid, block_dim=block)
      ctx.synchronize()

    with Timer("math_kernel_2"):
      # Should not have branch divergence
      ctx.enqueue_function[math_kernel_1](d_C, grid_dim=grid, block_dim=block)
      ctx.synchronize()

    with Timer("math_kernel_3"):
      # Should have branch divergence
      ctx.enqueue_function[math_kernel_1](d_C, grid_dim=grid, block_dim=block)
      ctx.synchronize()
