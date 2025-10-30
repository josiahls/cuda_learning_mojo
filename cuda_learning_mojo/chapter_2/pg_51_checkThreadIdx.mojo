from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext, Dim
from gpu.host.info import GPUInfo, _accelerator_arch
import sys


alias Int32Ptr = UnsafePointer[Int32]


fn initialize_int(ip: Int32Ptr, size: Int):
    for i in range(size):
        ip[i] = Int32(i)

fn print_matrix[nx:Int,ny:Int](mut C: Int32Ptr) raises:
    # We have a separate pointer that modify the offsets to.
    var ic = C
    print("\nMatrix ({}, {})".format(nx,ny))
    @parameter
    for iy in range(ny):
        @parameter
        for ix in range(nx):
            var s = String(ic[ix])
            if len(s) == 1:
                print("  ", s, sep='', end='')
            elif len(s) == 2:
                print(" ", s, sep='', end='')
            elif len(s) == 3:
                print(s, sep='', end='')
        ic += nx
        print()
    print()


fn print_thread_idx[nx:Int, ny:Int](A:Int32Ptr):
    var ix: Int32 = thread_idx.x # + block_idx.x * block_dim.x
    var iy: Int32 = thread_idx.y + block_idx.y * block_dim.y
    var idx: Int32 = iy * nx + ix


    print(
        "thread_id: (",
        thread_idx.x, thread_idx.y,
        "), block_id: (",
        block_idx.x, block_idx.y,
        "), coord thread_id: (",
        ix,iy,
        "), global indx ",
        idx,
        " ival ",
        A[Int32(idx)]
    )


fn main() raises:
    print("{} Starting...".format(sys.argv()[0]))
    alias dev:Int = 0

    print('Accelerator arch: ', "75")
    
    # gpu_info: GPUInfo = GPUInfo.from_name["75"]()
    # print("Using Device ", gpu_info);

    with DeviceContext(dev) as ctx:
        alias nx: Int = 9
        alias ny: Int = 6
        alias nxy:Int = nx * ny
        # alias nBytes = nxy * sys.info.size_of[Float32]()
        # NOTE: Why did the book specify float as opposed to int? 
        # these are int pointers...
        alias nBytes: Int = nxy * sys.info.size_of[Int32.dtype]()

        var h_A: Int32Ptr = Int32Ptr.alloc(nxy)
        initialize_int(h_A, nxy)

        print_matrix[nx, ny](h_A)

        var d_MatA = ctx.enqueue_create_buffer[Int32.dtype](nxy)
        ctx.enqueue_copy(d_MatA, h_A)

        alias block = Dim(4, 2)
        alias grid = Dim(
            UInt((nx + block.x() - 1) // block.x()), 
            UInt((ny + block.y() - 1) // block.y())
        )

        ctx.enqueue_function[print_thread_idx[nx, ny]](
            d_MatA,
            grid_dim = grid,
            block_dim = block
        )
        ctx.synchronize()


