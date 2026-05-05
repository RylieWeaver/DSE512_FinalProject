# Running Examples

Activate your environment and add the repository to `PYTHONPATH`:

```bash
source /path/to/your/environment/bin/activate
cd */DSE512_FinalProject
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd examples/
```

Run the single-process training example (decrease model context length and model size if on less formidable hardware):

```bash
python train.py --context_len 2048 --model_dim 1024
```

Run the distributed data-parallel/sequence-parallel test (assumes you have 4 GPUs available):

```bash
torchrun --standalone --nproc_per_node=4 test_dpsp.py --data_parallel_size 2 --sequence_parallel_size 2
```

Run the distributed training example (assumes you have 4 GPUs available):

```bash
torchrun --standalone --nproc_per_node=4 train_distributed.py --data_parallel_size 2 --sequence_parallel_size 2 --context_len 2048 --model_dim 1024
```
