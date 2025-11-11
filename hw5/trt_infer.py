# trt_infer_time_fixed.py
import time
import numpy as np
import tensorrt as trt
from cuda import cudart  # pip install cuda-python

ENGINE_PATH = "model.plan"

MIN_SHAPES = {"input_ids": (1, 12), "attention_mask": (1, 12), "token_type_ids": (1, 12)}
OPT_SHAPES = {"input_ids": (1, 12), "attention_mask": (1, 12), "token_type_ids": (1, 12)}
MAX_SHAPES = {"input_ids": (1, 12), "attention_mask": (1, 12), "token_type_ids": (1, 12)}

WARMUP = 10
N = 100

def npdtype(trt_dtype: trt.DataType):
    m = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF:  np.float16,
        trt.DataType.INT8:  np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL:  np.bool_,
    }
    # 兼容可能存在的 INT64 / UINT8
    m[getattr(trt.DataType, "INT64", trt.DataType.INT32)] = np.int64
    if hasattr(trt.DataType, "UINT8"):
        m[trt.DataType.UINT8] = np.uint8
    return m[trt_dtype]

def main():
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    with open(ENGINE_PATH, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # 列出所有绑定，便于核对名字
    n_bind = engine.num_bindings
    names   = [engine.get_binding_name(i) for i in range(n_bind)]
    is_in   = [engine.binding_is_input(i) for i in range(n_bind)]
    dtypes  = [engine.get_binding_dtype(i) for i in range(n_bind)]
    print("Bindings:", [f"{'[IN]' if is_in[i] else '[OUT]'} {names[i]}" for i in range(n_bind)])

    # 为存在的输入设置动态 shape（使用 OPT_SHAPES）
    for name, shp in OPT_SHAPES.items():
        idx = engine.get_binding_index(name)
        if idx == -1:
            # 该名字在 engine 中不存在，跳过
            continue
        if engine.binding_is_input(idx):
            context.set_binding_shape(idx, shp)

    # 分配 host/device 缓冲
    host_bufs = [None] * n_bind
    dev_bufs  = [None] * n_bind
    _, stream = cudart.cudaStreamCreate()

    for i in range(n_bind):
        # 对于输入，set_binding_shape 之后才能拿到具体 shape
        shape = tuple(context.get_binding_shape(i))
        dtype = npdtype(dtypes[i])

        if is_in[i]:
            nm = names[i]
            if nm == "input_ids":
                arr = np.random.randint(0, 30522, size=shape, dtype=np.int32).astype(dtype, copy=False)
            elif nm == "attention_mask":
                arr = np.ones(shape, dtype=np.int32).astype(dtype, copy=False)
            elif nm == "token_type_ids":
                arr = np.zeros(shape, dtype=np.int32).astype(dtype, copy=False)
            else:
                # 其他命名：用零填充
                arr = np.zeros(shape, dtype=dtype)
        else:
            arr = np.empty(shape, dtype=dtype)

        host_bufs[i] = arr
        _, dptr = cudart.cudaMalloc(arr.nbytes)
        dev_bufs[i] = dptr

    def infer_once():
        # H2D
        for i in range(n_bind):
            if is_in[i]:
                cudart.cudaMemcpyAsync(dev_bufs[i], host_bufs[i].ctypes.data, host_bufs[i].nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        # 执行
        ok = context.execute_async_v2(bindings=dev_bufs, stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")
        # D2H
        for i in range(n_bind):
            if not is_in[i]:
                cudart.cudaMemcpyAsync(host_bufs[i].ctypes.data, dev_bufs[i], host_bufs[i].nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

    # 预热
    for _ in range(WARMUP):
        infer_once()

    # 正式计时
    t0 = time.perf_counter()
    for _ in range(N):
        infer_once()
    t1 = time.perf_counter()
    Ttrt_ms = (t1 - t0) / N * 1000.0

    # 打印
    for i in range(n_bind):
        role = "IN " if is_in[i] else "OUT"
        print(f"[{role}] {names[i]:20s} shape={tuple(host_bufs[i].shape)} dtype={host_bufs[i].dtype}")
    print(f"Ttrt (avg over {N} runs): {Ttrt_ms:.3f} ms")

    # 释放
    for d in dev_bufs:
        if d: cudart.cudaFree(d)
    cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    main()