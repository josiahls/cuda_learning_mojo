from sys.info import size_of
from random import rand, seed
from gpu.host import DeviceContext, get_gpu_target, Dim
from gpu import thread_idx, block_idx, block_dim, grid_dim


fn check_index():
    print(
        "threadIdx: (",
        thread_idx.x,
        thread_idx.y,
        thread_idx.z,
        "), blockIdx: (",
        block_idx.x,
        block_idx.y,
        block_idx.z,
        "), blockDim: (",
        block_dim.x,
        block_dim.y,
        block_dim.z,
        "), gridDim: (",
        grid_dim.x,
        grid_dim.y,
        grid_dim.z,
        ")"
    )


fn main() raises:
    var n_elem: Int = 6
    var block: Dim = 3
    var grid: Dim = Int((n_elem + block.x() - 1) / block.x())


    print("grid.x", grid.x(), "grid.y", grid.y(), "grid.z", grid.z())
    print("block.x", block.x(), "block.y", block.y(), "block.z", block.z())

    # Initialize the device values
    with DeviceContext(0) as ctx:
        ctx.enqueue_function[check_index](
            grid_dim=grid, 
            block_dim=block
        )
        ctx.synchronize()


