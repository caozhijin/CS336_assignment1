import torch
import argparse
import os
from tqdm import trange
from cs336_basics.train import *
from cs336_basics.transformer_lm import Transformer_lm
import wandb


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 多GPU支持
    n_gpus = min(torch.cuda.device_count(), 3)   

    # 初始化 wandb
    wandb.init(project='cs336-bpe-transformer', config=vars(args))

    # Load datasets using memory mapping
    train_data = np.load(args.train_data_path, mmap_mode='r')
    val_data = np.load(args.val_data_path, mmap_mode='r')

    model = Transformer_lm(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.theta
    )
    if device == 'cuda' and n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model = model.to(device)
    optimizer = adamw(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # Load from checkpoint if available
    start_iter = 0
    if args.load_ckpt == 1:
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
            print(f"Resumed from checkpoint at iteration {start_iter}")

    progress_bar = trange(start_iter, args.total_iters, desc="Training")
    # early stop parameters
    best_val_loss = float('inf')
    no_improve_steps = 0

    for t in progress_bar:
        model.train()
        # Learning rate schedule
        lr_t = learning_rate_schedule(t, args.max_lr, args.min_lr, args.warmup_iters, args.total_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr_t

        optimizer.zero_grad()

        x, y = data_loading(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        loss.backward()
        #gradient clipping
        gradient_clipping(model.parameters(), args.max_grad_norm)

        optimizer.step()

        # Logging to progress bar & wandb
        if t % args.log_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=lr_t)
            wandb.log({
                'train_loss': loss.item(),
                'lr_t': lr_t,
                'batch_size': args.batch_size,
                'max_lr': args.max_lr,
                'min_lr': args.min_lr,
                'weight_decay': args.weight_decay,
                'warmup_iters': args.warmup_iters,
                'total_iters': args.total_iters,
                'context_length': args.context_length
            }, step=t)

        # Evaluation
        if t % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = data_loading(val_data, args.batch_size, args.context_length, device)
                val_logits = model(x_val)
                val_loss = cross_entropy(val_logits, y_val)
                print(f"[Eval @ Iter {t}] Val loss {val_loss.item():.4f}")
                wandb.log({'val_loss': val_loss.item(),
                           'lr_t': lr_t,
                            'batch_size': args.batch_size,
                            'max_lr': args.max_lr,
                            'min_lr': args.min_lr,
                            'weight_decay': args.weight_decay,
                            'warmup_iters': args.warmup_iters,
                            'total_iters': args.total_iters,
                            'context_length': args.context_length
                           }, step=t)
                
                # 早停机制
                early_stop_patience = 5 * args.eval_interval
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_steps = 0
                    # 只在val_loss变好时保存ckpt
                    if args.checkpoint_path:
                        ckpt_name = f"checkpoint_{t}_val{val_loss:.4f}.pt"
                        ckpt_path = os.path.join(args.checkpoint_path, ckpt_name)
                        save_checkpoint(model, optimizer, t, ckpt_path)
                        print(f"[Checkpoint @ Iter {t}] Saved to {ckpt_path}")
                else:
                    no_improve_steps += args.eval_interval
                    if no_improve_steps >= early_stop_patience:
                        print(f"Early stopping at iter {t}: val_loss has not improved for {early_stop_patience} steps.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='tinystories_result/train_encoded.npy')
    parser.add_argument('--val_data_path', type=str, default='tinystories_result/valid_encoded.npy')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint')

    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)

    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000)

    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--total_iters', type=int, default=5000)

    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=50)

    parser.add_argument('--load_ckpt', type = bool, default=0)
    args = parser.parse_args()
    
    train(args)
