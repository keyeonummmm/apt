import time

out_dir = 'out-apt'
eval_interval = 7
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2' # GPT-2 124M

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# Adjusted for dataset with ~945k training tokens
# Current setup: 32,768 tokens/iter means ~29 iters/epoch
# Modified for 3-5 epochs of training
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 100 # ~3.5 epochs

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False