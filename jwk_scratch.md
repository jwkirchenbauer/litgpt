initial testing

```
srun --pty --job-name=test_litgpt --partition=tron --qos=high -N1 --ntasks-per-node=4 --mem=50G --cpus-per-task=2 --gres=gpu:rtxa4000:4 --time=4:00:00 bash

ml cuda/12.9.1
conda_activate $WRKSPC/nexus_28_stable_litgpt


litgpt download EleutherAI/pythia-14m
litgpt download QwenQwen3-0.6B
litgpt download Qwen/Qwen3-0.6B-Base
litgpt download Qwen/Qwen2.5-0.5B
litgpt download Qwen/Qwen2.5-1.5B
litgpt download Qwen/Qwen2.5-7B
litgpt download meta-llama/Meta-Llama-3-8B

# 1 gpu (a4000)
srun -N1 -n1 --ntasks-per-node=1 --gpus-per-task=1 --mem=50G --cpus-per-task=2 --unbuffered \
    litgpt pretrain \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-0.5B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-0.5B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=1 \
# 4 gpus (a4000)
srun -N1 -n4 --ntasks-per-node=4 --mem=50G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-0.5B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-0.5B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=4 \
# 2  (h100)
srun -N1 -n2 --ntasks-per-node=2 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-1.5B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-1.5B \
    --train.micro_batch_size=4 \
    --train.global_batch_size=8 \
srun -N1 -n2 --ntasks-per-node=2 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-7B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-7B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=2 \
    --train.max_seq_length=2048 \

# 1  (h100)
srun -N1 -n1 --ntasks-per-node=1 --gpus-per-task=1 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-7B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-7B \
    --train.micro_batch_size=1 \
    --train.global_batch_size=1 \
    --train.max_seq_length=2048 \


# 4  (h100)
srun -N1 -n4 --ntasks-per-node=4 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Qwen2.5-7B \
    --tokenizer_dir=checkpoints/Qwen/Qwen2.5-7B \
    --train.micro_batch_size=2 \
    --train.global_batch_size=8 \
    --train.max_seq_length=4096 \

8192 toks / 0.770 s/step = 10.6k tps/gpu



# 4  (h100)
srun -N1 -n4 --ntasks-per-node=4 --mem=128G --cpus-per-task=2 --unbuffered \
    python -u litgpt/pretrain.py \
    --config=config_hub/pretrain/debug.yaml \
    --model_name=Llama-3-8B \
    --tokenizer_dir=checkpoints/meta-llama/Meta-Llama-3-8B \
    --train.micro_batch_size=2 \
    --train.global_batch_size=8 \
    --train.max_seq_length=4096 \

8192 toks / 0.830 s/step = 9.8k tps/gpu
at alloc/reserv 62gb/75gb of 80 so 77%/93%

note that early it does it 10.1k tps/gpu 
```

note that some cache related env vars are expected to be managed as you run out of space fast
```
# torch compile
export TRITON_CACHE_DIR="/cmlscratch/jkirchen/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/cmlscratch/jkirchen/.cache/inductor"

# general temp
export TMPDIR="/cmlscratch/jkirchen/.cache/tmp"
export TEMP=$TMPDIR
export TMP=$TMPDIR
```

issues:
- make sure the slurm procs and gpus launching is correct, its diff than usual
- https://github.com/Lightning-AI/litData/issues/482 suggest issue with the pretrain tutorial litdata usage
- Qwen3-0.6B-Base threw a cuda error on various bsz. Assumed it was vocab/tokenization, but did pass the correct tokenizer dir
- trying Qwen2.5-0.5B ... 
- okay nevermind cli args were wack nd wrong model was being loaded
- a4000s are so damn slow I forgot lol
- on 2xH100 we get 30k tps/gpu for the Qwen 1.5B, and  BAD?   for the 7B
- okay bf16 gives more reasonable thorughput on 7B like 8k. this could explain the original gap between lingua and litgptdev tbh, but will take more testing
against lingua in parallel



