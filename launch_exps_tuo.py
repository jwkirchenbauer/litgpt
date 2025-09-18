# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

WRITE_ONLY = True
# WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = (
    "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"
)

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# EXTRA_COMPILE_FLAGS = False
EXTRA_COMPILE_FLAGS = True

# LOG_RECOMPILES=False
LOG_RECOMPILES = True

QOS = "pdebug"
# QOS = "pbatch"
BANK = "effml"
TIME_LIMIT = 59
REPETITIONS = 1
DEPENDENCY = None

# QOS = "pbatch"
# BANK = "effml"
# # BANK = "guard"
# TIME_LIMIT = 1440
# REPETITIONS = 5
# DEPENDENCY = "afterany"

BASE_OUT_DIR = f"/p/vast1/kirchenb/singleshot-root/litgpt/outputs"

BASE_RUN_NAME = f"test"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

# INDUCTOR_CACHE=None
INDUCTOR_CACHE="/l/ssd/$USER"

GPN = 4

# Cfgs
exp_list = [
    # ["litgpt/pretrain.py", "config_hub/pretrain/debug.yaml", "Llama-3-8B", "checkpoints/meta-llama/Meta-Llama-3-8B", 1, GPN, 2, 8, 4096],
    # ["litgpt/pretrain.py", "config_hub/pretrain/debug.yaml", "Meta-Llama-3.1-8B", "checkpoints/meta-llama/Meta-Llama-3.1-8B", "checkpoints/meta-llama/Meta-Llama-3.1-8B", 1, GPN, 2, 8, 4096],
    ["litgpt/pretrain.py", "config_hub/pretrain/debug.yaml", "Meta-Llama-3.1-8B", "checkpoints/meta-llama/Meta-Llama-3.1-8B", "checkpoints/meta-llama/Meta-Llama-3.1-8B", 1, GPN, 1, 4, 4096],
]


# sweep_hparam = [
#     [
#     "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2/tiny_train_*.bin",
#     "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2/tiny_validation_*.bin",
#     473992006, # toks in train
#     ],
# ]
# exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cfg,
        model_name,
        tok_dir,
        init_ckpt_dir,
        nodes,
        gpn,
        mbsz,
        wbsz,
        slen,
        
    ) = exp

    gpus = nodes * gpn

    cli_args = ""

    cfg_str = cfg.split("/")[-1].replace(".yaml","")
    cli_args += f" --config={cfg}"

    model_str = model_name
    cli_args += f" --model_name={model_name} --tokenizer_dir={tok_dir} --initial_checkpoint_dir={init_ckpt_dir}"

    bsz_str = f"mb{mbsz}-wb{wbsz}-sl{slen}"
    cli_args += f" --train.micro_batch_size={mbsz} --train.global_batch_size={wbsz} --train.max_seq_length={slen}"

    # mod more things
    # ...

    # join to a unique run name for the experiment
    run_name = (
        f"{BASE_RUN_NAME}_{cfg_str}_{model_str}_{bsz_str}_{nodes}N{gpus}n"
    )

    # put together the actual "train.py" command
    custom_invocation = f"python -u {script} {cli_args}"

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
        --wandb_offline={WANDB_OFFLINE} \
        --rocm_version={ROCM_VERSION} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --rccl_cfg={RCCL_CFG} \
        --cache_dir={INDUCTOR_CACHE} \
        --qos={QOS} \
        --bank={BANK} \
        --repetitions={REPETITIONS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
        --minutes={TIME_LIMIT} \
        --nodes={nodes} \
        --gpus_per_node={gpn} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation}' \
        --pass_run_name=False \
        --add_compile_flags={EXTRA_COMPILE_FLAGS} \
        --log_recompiles={LOG_RECOMPILES} \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)

print(f"Total launches: {total_launches}")
