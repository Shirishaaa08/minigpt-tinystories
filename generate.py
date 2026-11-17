import argparse
import os
import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import load_bpe_tokenizer, EOT_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="Generate stories with MiniGPT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/minigpt_best.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="checkpoints/tokenizer", help="Directory with tokenizer files")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily.", help="Initial prompt")
    parser.add_argument("--max_tokens", type=int, default=250, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = safer, higher = creative)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling filter")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt loop")
    return parser.parse_args()


@torch.no_grad()
def generate_story(
    model: MiniGPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 250,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu",
    eot_id: int = 0,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Crop context to model's maximum block size
        x_cond = x[:, -model.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :].float() / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
            logits[logits < kth] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eot_id:
            break

        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}. Please train the model first.")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config", {
        "vocab_size": 4096,
        "block_size": 256,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.0,
    })

    model = MiniGPT(**config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded successfully.")

    tokenizer = load_bpe_tokenizer(args.tokenizer_dir)
    eot_id = tokenizer.token_to_id(EOT_TOKEN)

    if args.interactive:
        print("\n=== MiniGPT Story Generator (Type 'exit' to quit) ===")
        while True:
            prompt = input("\nEnter prompt: ").strip()
            if not prompt or prompt.lower() in ("exit", "quit"):
                break
            print("\nGenerating...\n" + "-" * 50)
            story = generate_story(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
                eot_id=eot_id
            )
            print(story)
            print("-" * 50)
    else:
        print(f"\nPrompt: {args.prompt}\n" + "-" * 50)
        story = generate_story(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            eot_id=eot_id
        )
        print(story)
        print("-" * 50)


if __name__ == "__main__":
    main()
