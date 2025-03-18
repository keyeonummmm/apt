import os
import torch
import pickle
from contextlib import nullcontext
import tiktoken
from model import GPTConfig, GPT
import signal
import time
import sys

init_from ='resume' # resume from an out_dir, or a gpt2 variant (e.g. 'gpt-2')
out_dir = 'finetuning/out-apt' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 150 # number of samples to draw
max_new_tokens = 200 # number of tokens generated in each sample
temperature = 1 # less random < 1 < more random
top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337 # random seed for sampling
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
dtype = 'bfloat16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16' # 'float16', 'bfloat16', or 'float32'
compile = False # use PyTorch 2.0 to compile the model to be faster
# Get directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Load configurator.py from same directory as this script
exec(open(os.path.join(script_dir, 'configurator.py')).read())

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ptdtype = {'float32' : torch.float32,
            'bfloat16' : torch.bfloat16,
            'float16' : torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if init_from == 'resume':
    project_root = os.path.dirname(script_dir)
    ckpt_path = os.path.join(project_root, out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device) #added weights_only=True for security
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefixes = '_orig_mod'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefixes):
            state_dict[k[len(unwanted_prefixes):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) #requires PyTorch 2.0

load_meta = False
if init_from == 'resume':
    meta_path = os.path.join('data', 'one', 'meta.pkl')  # Hardcode to 'one' dataset or modify as needed
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Add signal handler before model generation
def signal_handler(sig, frame):
    print('\nExiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Replace the final generation loop with:
# print("Generating output every 5 seconds. Press Ctrl+C to exit.")
with torch.no_grad():
    with ctx:
        while True:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('------------------------------------------------------------')
            # time.sleep(5)