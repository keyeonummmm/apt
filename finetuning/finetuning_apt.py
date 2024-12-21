import os
import time
from dataclasses import dataclass
from contextlib import nullcontext
import wandb
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPT

#Model architecture
class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

#Finetuning parameters
out_dir = 'out-apt'
eval_interval = 20
eval_iters = 40
log_interval = 1
eval_only = False
wandb_log = False
wandb_project = 'apt'
wandb_run_name = 'ft-' + str(time.time())
dataset = 'apt'
batch_size = 1
init_from = 'gpt2'
always_save_checkpoint = False
# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# Adjusted for dataset with ~945k training tokens
# Current setup: 32,768 tokens/iter means ~29 iters/epoch
# Modified for 3-5 epochs of training
gradient_accumulation_steps = 32
max_iters = 100 # ~3.5 epochs
learning_rate = 3e-5 # finetune at constant LR
decay_lr = True
warmup_iters = 50
lr_decay_iters = 250
min_lr = learning_rate / 10
grad_clip = 1.0

#Model parameters
block_size = 1024
dropout = 0.0
bias = True

# System/device config
backend = 'nccl'
def get_device_config():
    if torch.cuda.is_available():
        return 'cuda', 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16', True
    elif torch.backends.mps.is_available():
        return 'mps', 'float32', False
    else:
        return 'cpu', 'float32', False

device, dtype, compile = get_device_config()

# DDP settings (CUDA-specific)
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Device-specific configurations
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Various inits
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"training {tokens_per_iter:,} tokens per iteration")
    os.makedirs(out_dir, exist_ok=True)
    
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#Data loading
data_dir = 'data'
def get_batch(split):
    #Recreate memmap each time to prevent memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    #CUDA-specific optimizations
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

#Model init
def ensure_model_downloaded(model_name):
    """Check if model exists in cache, download only if needed."""
    from transformers import GPT2LMHeadModel
    
    try:
        # from_pretrained handles caching automatically
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            local_files_only=True
        )
        print(f"Found {model_name} in cache")
        return model
    except Exception:
        print(f"Downloading {model_name}...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print(f"Successfully downloaded {model_name}")
        return model

print(f"Ensuring {init_from} model is available...")
try:
    downloaded_model = ensure_model_downloaded(init_from)
except Exception as e:
    print(f"Failed to download model: {str(e)}")
    raise

model = GPT.from_pretrained(init_from, dict(dropout=dropout))
model = model.to(device)
model_args = dict(n_layer=model.config.n_layer,
                   n_head=model.config.n_head,
                   n_embd=model.config.n_embd,
                   block_size=model.config.block_size,
                   bias=model.config.bias,
                   vocab_size=model.config.vocab_size)
config = model.config

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, 
                                    betas=(0.9, 0.95), device_type=device_type)
if device_type == 'cuda':
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
else:
    scaler = torch.cuda.amp.GradScaler(enabled=False)

if compile and 'cuda' in device_type:
    print("Compiling model...")
    model = torch.compile(model)
else:
    print(f'Model compilation disabled for {device_type}')

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

def get_lr(iter_num):
    # Linear warmup
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # Cosine decay
    if iter_num > lr_decay_iters:
        return min_lr
    # In between, use cosine decay
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize tracking variables before training loop
best_val_loss = float('inf')
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

def save_checkpoint():
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    # Also save as latest if it's the best so far
    if losses['val'] < best_val_loss:
        torch.save(checkpoint, os.path.join(out_dir, 'best_ckpt.pt'))

# Training loop
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses['val'])
            save_checkpoint()

    if iter_num == 0 and eval_only:
        break

    # forward backward update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()