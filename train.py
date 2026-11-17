import argparse
import math
import os
import time
from contextlib import nullcontext
import numpy as np
import torch

from model import MiniGPT
from tokenizer import train_bpe_tokenizer, load_bpe_tokenizer, EOT_TOKEN
from dataset import stream_tinystories, tokenize_and_cache, get_batch

def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniGPT on TinyStories")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--n_train_stories", type=int, default=500_000, help="Number of training stories to stream")
    parser.add_argument("--n_val_stories", type=int, default=2_000, help="Number of validation stories")
    parser.add_argument("--vocab_size", type=int, default=4096, help="BPE vocabulary size")
    
    # Model architecture
    parser.add_argument("--block_size", type=int, default=256, help="Context length in tokens")
    parser.add_argument("--n_layer", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--log_every", type=int, default=25, help="Logging interval")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=300, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Note: Running on CPU will be slow. A GPU (such as Nvidia T4/A100) is strongly recommended.")

    # 1. Tokenizer setup
    tok_dir = os.path.join(args.out_dir, "tokenizer")
    if os.path.exists(os.path.join(tok_dir, "vocab.json")):
        tokenizer = load_bpe_tokenizer(tok_dir)
    else:
        train_stories = stream_tinystories(split="train", n_samples=min(100_000, args.n_train_stories))
        tokenizer = train_bpe_tokenizer(
            train_stories,
            vocab_size=args.vocab_size,
            min_frequency=2,
            save_dir=tok_dir
        )
        del train_stories

    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {vocab_size} | EOT ID: {eot_id}")

    # 2. Data preparation
    train_cache = os.path.join(args.out_dir, "train_ids.npy")
    val_cache = os.path.join(args.out_dir, "val_ids.npy")

    if not os.path.exists(train_cache):
        raw_train = stream_tinystories(split="train", n_samples=args.n_train_stories)
        train_ids = tokenize_and_cache(raw_train, tokenizer, eot_id, train_cache)
        del raw_train
    else:
        train_ids = np.load(train_cache)

    if not os.path.exists(val_cache):
        raw_val = stream_tinystories(split="validation", n_samples=args.n_val_stories)
        val_ids = tokenize_and_cache(raw_val, tokenizer, eot_id, val_cache)
        del raw_val
    else:
        val_ids = np.load(val_cache)

    print(f"Train tokens: {len(train_ids):,} | Validation tokens: {len(val_ids):,}")

    # 3. Model instantiation
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Expected starting loss: ~ln({vocab_size}) = {math.log(vocab_size):.2f}")

    # 4. Optimizer & Mixed Precision
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    use_cuda = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_cuda else nullcontext()

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)

    @torch.no_grad()
    def estimate_loss(n_batches: int = 30):
        model.eval()
        out = {}
        for split, data in [("train", train_ids), ("val", val_ids)]:
            losses = []
            for _ in range(n_batches):
                x, y = get_batch(data, args.block_size, args.batch_size, device)
                with amp_ctx:
                    _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # 5. Checkpoint Resume
    ckpt_path = os.path.join(args.out_dir, "minigpt_best.pt")
    start_step = 0
    best_val = float("inf")
    history = {"step": [], "train": [], "val": []}

    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint {ckpt_path}...")
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        best_val = ck["best_val"]
        history = ck.get("history", history)
        print(f"Resumed from step {ck['step']} with best val loss {best_val:.3f}")

    # 6. Training Loop
    print("\n--- Starting Training ---")
    print(f"Tokens per step: {args.batch_size * args.block_size:,}")
    model.train()
    t0 = time.time()
    t_log = time.time()

    for step in range(start_step, args.max_steps + 1):
        lr_now = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        # Periodic Evaluation
        if step % args.eval_every == 0 or step == args.max_steps:
            l = estimate_loss()
            history["step"].append(step)
            history["train"].append(l["train"])
            history["val"].append(l["val"])
            elapsed = time.time() - t0
            print(f"\n[Step {step:>5}/{args.max_steps}] Train Loss: {l['train']:.3f} | Val Loss: {l['val']:.3f} | Elapsed: {elapsed:5.0f}s")

            if l["val"] < best_val:
                best_val = l["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_val": best_val,
                        "history": history,
                        "config": {
                            "vocab_size": vocab_size,
                            "block_size": args.block_size,
                            "n_layer": args.n_layer,
                            "n_head": args.n_head,
                            "n_embd": args.n_embd,
                            "dropout": args.dropout,
                        },
                    },
                    ckpt_path,
                )
                print(f"  -> Best checkpoint saved to {ckpt_path}")

        if step == args.max_steps:
            break

        # Forward & Backward Pass
        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        with amp_ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Periodic logging
        if step % args.log_every == 0 and step > start_step:
            if use_cuda:
                torch.cuda.synchronize()
            dt = (time.time() - t_log) / args.log_every
            eta = dt * (args.max_steps - step) / 60
            tok_per_sec = (args.batch_size * args.block_size) / dt
            print(
                f"  step {step:>5} | loss {loss.item():.3f} | lr {lr_now:.2e} | "
                f"{dt:.2f}s/step | {tok_per_sec:,.0f} tok/s | ETA {eta:.1f} min"
            )
            t_log = time.time()

    print(f"\nTraining completed! Best validation loss: {best_val:.3f}")


if __name__ == "__main__":
    main()
