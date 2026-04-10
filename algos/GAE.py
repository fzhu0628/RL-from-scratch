import torch

def gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    gae = 0
    advantages = []
    values_ext = torch.cat([values, next_value.unsqueeze(0)])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t+1] - values_ext[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.stack(advantages)