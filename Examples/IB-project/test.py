import multiprocessing.shared_memory
import numpy as np
import multiprocessing
import time


# 打开共享内存
shm_name = "pos"
shm_size = 1024 * 1024 * 10  # 10MB
shm = multiprocessing.shared_memory.SharedMemory(name=shm_name,create=True, size=shm_size)

# 动态创建数组
data = np.random.rand(10, 3).astype(np.float32)

def WriteIntoMemory(data,mem):
    # 将元信息和数据写入共享内存
    meta_info = np.array(data.shape + (data.dtype.itemsize,), dtype=np.int32)
    meta_size = meta_info.nbytes
    data_size = data.nbytes

    meta_array = np.ndarray(meta_info.shape, dtype=np.int32, buffer=mem.buf[:meta_size])
    np.copyto(meta_array, meta_info)

    #
    data_array = np.ndarray(data.shape, dtype=data.dtype, buffer=mem.buf[meta_size:meta_size + data_size])
    np.copyto(data_array, data)

WriteIntoMemory(data,shm)

print("Taichi data written to shared memory.")
print(data)
# 保持共享内存直到进程结束
time.sleep(60)
shm.close()
# shm.unlink()
# print("Shared memory unlinked.")

