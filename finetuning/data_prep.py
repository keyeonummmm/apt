import os
import requests
import tiktoken
import numpy as np

# Create data1 directory in APT root
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

# Load the dataset
input_file_path = os.path.join(os.path.dirname(__file__), '/Users/zhouchaoran/Desktop/APT/one.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/keyeonummmm/apt/master/one.txt'
    with open(input_file_path, 'w', encoding='UTF-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='UTF-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding('gpt2')
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files in data1 directory
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_path = os.path.join(data_dir, 'train.bin')
val_path = os.path.join(data_dir, 'val.bin')

train_ids.tofile(train_path)
val_ids.tofile(val_path)

print(f"\nFiles saved in {data_dir}:")
print(f"- {os.path.basename(train_path)}")
print(f"- {os.path.basename(val_path)}")

#train has 955,986 tokens
#val has 110,254 tokens