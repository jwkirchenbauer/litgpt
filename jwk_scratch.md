initial testing

```
salloc --job-name=test_litgpt --partition=tron --qos=high -N1 --ntasks-per-node=4 --mem=50G --cpus-per-task=2 --gres=gpu:rtxa4000:4 --time=4:00:00

ml cuda/12.9.1
conda_activate $WRKSPC/nexus_28_stable_litgpt


litgpt download EleutherAI/pythia-14m
litgpt download QwenQwen3-0.6B
litgpt download Qwen/Qwen3-0.6B-Base

srun -N1 -n4 --ntasks-per-node=4 --mem=50G --cpus-per-task=2 \
    litgpt pretrain Qwen3-0.6B-Base \
    --config=config_hub/pretrain/debug.yaml \
    --tokenizer_dir=checkpoints/Qwen/Qwen3-0.6B-Base \
    --train.micro_batch_size=4 \
    --train.global_batch_size=16 \

```

issues:
- make sure the slurm procs and gpus launching is correct, its diff than usual
- https://github.com/Lightning-AI/litData/issues/482 suggest issue with the pretrain tutorial litdata usage