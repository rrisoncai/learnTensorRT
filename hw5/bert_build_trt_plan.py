# build_engine.py
import tensorrt as trt

ONNX_PATH = "bert-base-uncased/model-sim.onnx"
ENGINE_PATH = "model.plan"
# 根据你的设备与需求调整
MIN_SHAPES = {"input_ids": (1, 12), "attention_mask": (1, 12), "token_type_ids": (1, 12)}
OPT_SHAPES = {"input_ids": (1,12), "attention_mask": (1,12), "token_type_ids": (1,12)}
MAX_SHAPES = {"input_ids": (1,12), "attention_mask": (1,12), "token_type_ids": (1,12)}
USE_FP16 = True
WORKSPACE_MB = 4096

def build():
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")  # 如需插件，这行能初始化内置插件
    with trt.Builder(logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser, \
         builder.create_builder_config() as config:

        # 读取 ONNX
        with open(ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise SystemExit("ONNX 解析失败")

        # workspace
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1<<20))
        # 精度
        if USE_FP16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # 优化 profile（动态 shape 必须）
        profile = builder.create_optimization_profile()
        for name, shape in MIN_SHAPES.items():
            if network.get_input(0).shape[0] == -1 or True:  # 显式设置所有输入
                profile.set_shape(name, MIN_SHAPES[name], OPT_SHAPES[name], MAX_SHAPES[name])
        config.add_optimization_profile(profile)

        # 构建
        engine = builder.build_engine(network, config)
        if engine is None:
            raise SystemExit("Engine 构建失败")

        # 序列化保存
        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())
        print(f"Saved TensorRT engine to {ENGINE_PATH}")

if __name__ == "__main__":
    build()