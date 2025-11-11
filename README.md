# learnTensorRT

## Environment Setup in AutoDL

GPU:
- vGPU 32GB
- 3090

using docker image
- pytorch=2.0.0
- cuda=11.8
- python=3.8 (ubuntu20.04)

```bash
conda init base
source /etc/network_turbo
apt update
apt install -y git-lfs
git lfs install
cd hw5
git clone https://huggingface.co/google-bert/bert-base-uncased
pip install -i https://mirrors.aliyun.com/pypi/simple onnxruntime-gpu==1.18.1
pip install transformers
pip install cuda-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorrt==8.6.1 --extra-index-url https://pypi.nvidia.com/simple
```