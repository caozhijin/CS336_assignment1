from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import numpy as np
import numpy.typing as npt

def cross_entropy(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs (Float[Tensor, "batch_size seq_len vocab_size"]): inputs[i][j] is the
    #         unnormalized logit of jth class for the ith example.
    # targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size seq_len) with the index of the correct class.
    #         Each value must be between 0 and `num_classes - 1`.
    # Returns:
    #     Float[Tensor, ""]: The average cross-entropy loss across examples.
    log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True) 
    loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    return loss

class sgd_optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        super().__init__(params, {'lr': lr})

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    # optimizer always use .data and .grad.data to avoid tracking in autograd
                    p.data -= lr * p.grad.data
        return loss
    
class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data) #m
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data) #v
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() + group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
        return loss

def learning_rate_schedule(t:int, alpha_max:float, alpha_min:float, T_w:int, T_c:int) -> float:
    """
    Sets the learning rate of the optimizer according to the cosine annealing schedule.
    Args:
        t (int): The current epoch number.
        alpha_max (float): The maximum learning rate.
        alpha_min (float): The minimum learning rate.
        T_w (int): The number of warm-up epochs.
        T_c (int): The total number of epochs.
    Returns:
        float: The learning rate for the current epoch. 
    """
    if t < T_w:
        return alpha_max * t / T_w
    elif t <= T_c:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
    else:
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], M: float) -> None:
    """
    Clips the gradients of the given parameters to have a maximum norm of M.
    Args:
        M (float): The maximum allowed norm of the gradients.
        parameters (Iterable[torch.nn.Parameter]): The parameters whose gradients will be clipped.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > M:
        clip_coef = M / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads data from the given dataset and returns a batch of input and target tensors.
    Args:
        dataset (npt.NDArray): The dataset to load data from.
        batch_size (int): The number of samples in each batch.
        context_length (int): The length of the context window.
        device (str): The device to load the tensors onto.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
    """
    num_samples = dataset.shape[0] - context_length
    # from ok_to_use ids get batch_size random samples
    indices = np.random.choice(num_samples, batch_size, replace=False)
    x_batch = np.array([dataset[i:i+context_length] for i in indices], dtype=np.int64)
    y_batch = np.array([dataset[i+1:i+context_length+1] for i in indices], dtype=np.int64)
    x_tensor = torch.tensor(x_batch, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)
    return x_tensor, y_tensor

def save_checkpoint(model, optimizer, iteration, out):
    # model: torch.nn.Module
    # optimizer: torch.optim.Optimizer
    # iteration: int
    # out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer) -> int:
    # src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    # model: torch.nn.Module
    # optimizer: torch.optim.Optimizer
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration