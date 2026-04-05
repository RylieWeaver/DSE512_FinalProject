# Environment Setup

## DGX
```
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init DSE512 --bare --python 3.12 && cd DSE512
uv venv dse --python 3.12 --native-tls && source dse/bin/activate
export CUDA_HOME=/mnt/DGX01/Personal/$USER/Environments/CUDA/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
uv pip install ninja packaging psutil
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install numpy einops
uv pip install flash-attn --no-build-isolation
```

## ISAAC (still under development)
```
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=/place/with/storage

uv init DSE512 --bare --python 3.12 && cd DSE512
uv venv dse --python 3.12 --native-tls && source dse/bin/activate
uv pip install torch numpy einops
```


# Running Examples
```
cd */DSE512_FinalProject
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd examples/
python train.py --context_len 2048 --model_dim 1024
torchrun --standalone --nproc_per_node=4 test_dpsp.py  --data_parallel_size 2  --sequence_parallel_size 2
torchrun --standalone --nproc_per_node=4 train_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 --context_len 2048 --model_dim 1024
```
