import torch
from cs336_basics.softmax import Softmax

def decoding( model,tokenizer,
              prompt: str,
              max_new_tokens: int = 50,
              temperature: float = 1.0,
              top_p: float = 1.0,) -> str:
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)  # shape: (1, seq_len, vocab_size)
        logits = logits[:, -1, :]  # last token's logits: (1, vocab_size)

        # temperature scaling
        logits = logits / temperature
        softmax = Softmax(-1)
        probs = softmax(logits)

        # nucleus or top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff[...,1:] = cutoff[..., :-1]
        cutoff[0] = 0
        sorted_probs[cutoff] = 0
        sorted_probs_nor = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        #sample
        next_token = torch.multinomial(sorted_probs_nor, num_samples=1)
        next_token_id = sorted_indices.gather(-1, next_token)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.encode("<|endoftext|>")[0]:
            break
            
    return tokenizer.decode(input_ids[0].tolist())

