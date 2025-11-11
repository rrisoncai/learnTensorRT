# onnx_bert_infer_time.py
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = "bert-base-uncased/model.onnx"  # 你的模型路径
BATCH = 1
SEQ = 128
VOCAB_SIZE = 30522  # BERT base 的词表大小

# 优化与 EP 设置：优先 CUDA，回退 CPU
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print("Available providers:", sess.get_providers())

# 根据模型的输入名自动适配（常见：input_ids/attention_mask/token_type_ids）
need = {i.name for i in sess.get_inputs()}
inputs = {}

# 估计批次与序列长度（若模型给了动态维度，这里用默认 BATCH/SEQ）
try:
    s0 = sess.get_inputs()[0].shape
    b = BATCH if s0[0] is None else int(s0[0])
    l = SEQ  if s0[1] is None else int(s0[1])
except Exception:
    b, l = BATCH, SEQ

if "input_ids" in need:
    inputs["input_ids"] = np.random.randint(0, VOCAB_SIZE, size=(b, l), dtype=np.int64)
if "attention_mask" in need:
    inputs["attention_mask"] = np.ones((b, l), dtype=np.int64)
if "token_type_ids" in need:
    inputs["token_type_ids"] = np.zeros((b, l), dtype=np.int64)

# 若模型使用其它命名（非常少见），退化为“把第一个输入当 input_ids”
if not inputs:
    xname = sess.get_inputs()[0].name
    inputs[xname] = np.random.randint(0, VOCAB_SIZE, size=(b, l), dtype=np.int64)

# 预热（让内核加载/图优化等完成）
for _ in range(10):
    _ = sess.run(None, inputs)

# 正式计时
N = 100
t0 = time.perf_counter()
for _ in range(N):
    out = sess.run(None, inputs)   # 读取输出可确保计算完成
t1 = time.perf_counter()

Tort_ms = (t1 - t0) / N * 1000.0
print(f"Tort (avg over {N} runs): {Tort_ms:.3f} ms")
# 可选：看一下主输出的形状
print("Output[0] shape:", [o.shape for o in out])