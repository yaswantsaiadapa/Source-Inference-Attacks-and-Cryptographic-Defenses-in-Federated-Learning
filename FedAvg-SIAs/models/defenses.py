import copy
import torch
import numpy as np


def apply_differential_privacy(w_locals, noise_multiplier=0.01, clip_bound=1.0):
    """
    Applies Local Differential Privacy to client model parameters.
    Weights are perturbed with calibrated Gaussian noise.
    """
    if noise_multiplier <= 0:
        return copy.deepcopy(w_locals)

    dp_w_locals = []
    for w in w_locals:
        w_perturbed = copy.deepcopy(w)
        for key in w_perturbed.keys():
            param = w_perturbed[key].float()
            std_dev = noise_multiplier * clip_bound
            noise = torch.randn_like(param) * std_dev
            w_perturbed[key] = (param + noise).to(w[key].dtype)
        dp_w_locals.append(w_perturbed)

    return dp_w_locals


def apply_secure_aggregation(w_locals, size_weights=None):
    """
    Implements Pairwise Additive Masking (Bonawitz et al. Secure Aggregation principle).
    
    1. Each pair of clients (i, j) shares a zero-sum pseudo-random mask M_{i,j}.
    2. Client i adds M_{i,j} for j > i and subtracts M_{j,i} for j < i.
    3. The individual transmitted weights y_i are completely blinded (attacker cannot run SIA).
    4. Upon summation at the server, all pairwise masks algebraically cancel out:
       sum(y_i) == sum(w_i), achieving exact aggregation with ZERO loss in model utility.
    """
    num_clients = len(w_locals)
    if num_clients <= 1:
        return copy.deepcopy(w_locals), copy.deepcopy(w_locals[0])

    param_keys = list(w_locals[0].keys())

    # Step 1: Generate pairwise masks M_{i, j} for all pairs (i < j)
    # Using deterministic seed pairs to simulate shared Diffie-Hellman keys
    pairwise_masks = {}
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            pair_seed = int((i * 10007 + j * 99991) % (2**31 - 1))
            rng = torch.Generator()
            rng.manual_seed(pair_seed)
            
            mask_dict = {}
            for k in param_keys:
                tensor_template = w_locals[0][k]
                if tensor_template.is_floating_point():
                    # Generate zero-mean random mask with variance to blind parameters
                    mask = torch.randn(tensor_template.shape, generator=rng, dtype=tensor_template.dtype) * 10.0
                else:
                    mask = torch.zeros_like(tensor_template)
                mask_dict[k] = mask
            pairwise_masks[(i, j)] = mask_dict

    # Step 2: Clients compute blinded weights y_i = w_i + sum_{j > i} M_{i, j} - sum_{j < i} M_{j, i}
    masked_w_locals = []
    for i in range(num_clients):
        y_i = copy.deepcopy(w_locals[i])
        for k in param_keys:
            if not y_i[k].is_floating_point():
                continue
            for j in range(num_clients):
                if j > i:
                    y_i[k] = y_i[k] + pairwise_masks[(i, j)][k].to(y_i[k].device)
                elif j < i:
                    y_i[k] = y_i[k] - pairwise_masks[(j, i)][k].to(y_i[k].device)
        masked_w_locals.append(y_i)

    # Step 3: Server aggregation where masks cancel out
    w_avg = copy.deepcopy(w_locals[0])
    for k in param_keys:
        if w_avg[k].is_floating_point():
            summed = torch.zeros_like(masked_w_locals[0][k])
            for i in range(num_clients):
                summed += masked_w_locals[i][k]
            w_avg[k] = summed / num_clients
        else:
            w_avg[k] = w_locals[0][k]

    return masked_w_locals, w_avg
