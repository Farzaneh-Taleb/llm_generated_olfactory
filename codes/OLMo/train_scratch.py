import argparse, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class TokenDataset(Dataset):
    def __init__(self, path, seq_len):
        data = np.load(path)
        self.data = torch.from_numpy(data).long()
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.data[i:i+self.seq_len]
        y = self.data[i+1:i+self.seq_len+1]
        return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tokens_npy", required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=10000)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    dataset = TokenDataset(args.train_tokens_npy, args.seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # VERY simple transformer (≈100M-ish depending on dims)
    model = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072
        ),
        num_layers=12,
    )
    lm_head = nn.Linear(768, 50257)

    model = nn.Sequential(model, lm_head).to(device)
    model = DDP(model, device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    model.train()
    for x, y in loader:
        if step >= args.steps:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if rank == 0 and step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f}")
        step += 1

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
