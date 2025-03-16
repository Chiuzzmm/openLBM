import numpy as np
import multiprocessing
import multiprocessing.shared_memory
from time import sleep




def read_from_shared_memory(shm_name):
    # 打开共享内存
    shm = multiprocessing.shared_memory.SharedMemory(name=shm_name,create=False)

    # 读取元信息
    meta_size = 12  # 假设 meta_info 是一个 (3,) 的 int32 数组，占用 12 字节
    meta_array = np.ndarray((3,), dtype=np.int32, buffer=shm.buf[:meta_size])
    
    # 解析元信息
    shape = tuple(meta_array[:2])  # 前两个值为形状
    itemsize = meta_array[2]      # 最后一个值为数据类型大小

    # 根据元信息动态创建数组
    dtype = np.dtype(np.float32) if itemsize == 4 else np.dtype(np.float64)
    data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf[meta_size:])

    data_array=data_array.copy()
    return data_array

# 示例读取
shm_name = "pos"
data = read_from_shared_memory(shm_name)
print(f"Data shape: {data.shape}, Data type: {data.dtype}")

print(data)




