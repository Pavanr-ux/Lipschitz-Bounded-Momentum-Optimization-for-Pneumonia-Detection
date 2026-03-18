import torch

class LipschitzMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9):
        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                m = state['momentum']

                L = torch.norm(grad) + 1e-6
                dynamic_beta = beta / (1 + L)

                m.mul_(dynamic_beta).add_(grad)
                p.data -= lr * m