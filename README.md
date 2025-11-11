# learnTensorRT

## Environment Setup in AutoDL

using image
- pytorch=2.0.0
- cuda=11.8
- python=3.8 (ubuntu20.04)

```bash
conda init base
source /etc/network_turbo
apt update
apt install -y git-lfs
git lfs install
git clone https://huggingface.co/google-bert/bert-base-uncased
pip install -i https://mirrors.aliyun.com/pypi/simple onnxruntime-gpu==1.18.1
pip install transformers
```