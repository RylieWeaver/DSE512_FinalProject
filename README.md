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
uv pip install numpy einops pysam pandas tqdm
uv pip install flash-attn --no-build-isolation

mkdir -p bin && cd bin
curl -o datasets https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets
chmod +x datasets
export PATH=$(pwd):$PATH
```

## ISAAC (still under development)
```
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_CACHE_DIR=/place/with/storage

uv init DSE512 --bare --python 3.12 && cd DSE512
uv venv dse --python 3.12 --native-tls && source dse/bin/activate
uv pip install torch numpy einops pysam pandas tqdm
```

## Frontier
```
### Start Env ###
Install uv if not already: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init DSE512 --bare --python 3.13 && cd DSE512
uv venv dse --python 3.13 --native-tls && source dse/bin/activate
export UV_CACHE_DIR=$(pwd)


### Torch / Flash Attn ### (based on https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#flash-attention)
# Load modules
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Install Setup/Torch Packages #
uv pip install pip ninja packaging wheel
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1

# Because using a non-default CPE
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# Activate your virtual environment
source /lustre/orion/lrn036/world-shared/rylieweaver/Environments/DSE512/dse/bin/activate

# Retrieve the FA repo
git clone https://github.com/ROCm/flash-attention
cd flash-attention/
git checkout v2.8.4.1-cktile
git submodule init
git submodule update

# Build the flash-attn wheel (Triton Backend):
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python -m pip wheel . -w dist --no-build-isolation

# Install flash-attn
pip install dist/*.whl


### Install Rest of Packages ###
uv pip install numpy einops pysam pandas tqdm psutil matplotlib


### Install NCBI "Datasets" ###
mkdir -p bin && cd bin
curl -o datasets https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets
chmod +x datasets
export PATH=$(pwd):$PATH

### NOTES ###
When hipcc wasn't found I just re-ran module loads.
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
