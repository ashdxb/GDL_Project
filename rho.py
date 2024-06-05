import torch
from typing import Tuple

'''
    rho via RKHS:
'''
class ED(torch.autograd.Function):
    '''
        There is numerical instability in gradient computation of eigendecomposition:
            lambda_, U = torch.linalg.eigh(K)
        Thus we provide numerically stable backward.
    '''
    @staticmethod
    def forward(ctx, K:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lambda_, U = torch.linalg.eigh(K)
        ctx.save_for_backward(lambda_, U)
        return lambda_, U

    @staticmethod
    def backward(ctx, lambda_grad: torch.Tensor, U_grad: torch.Tensor) -> torch.Tensor:
        lambda_, U = ctx.saved_tensors
        I = torch.eye(lambda_.shape[0], device=lambda_.device)
        tmp = lambda_.reshape(-1, 1) - lambda_.reshape(1, -1) + I
        eps = 1e-5
        tmp = tmp + (tmp == 0) * eps # prevents nans
        K_tilde = 1/tmp - I
        return U @ (K_tilde.T * (U.T @ U_grad) + torch.diag(lambda_grad)) @ U.T


def Gau_kernel(A: torch.Tensor, B: torch.Tensor, lambda_:float=1.0) -> torch.Tensor:
    '''
        Gaussian kernel, which is universal according:
            https://jmlr.csail.mit.edu/papers/volume7/micchelli06a/micchelli06a.pdf

        A: tensor of shape (num_samples, dim)
        B: tensor of shape (num_samples, dim)
    '''
    A_reduced = (A * A).sum(dim=-1).reshape(-1, 1)  # column vector (num_samples, 1)
    B_reduced = (B * B).sum(dim=-1).reshape(1, -1)  # row vector (1, num_samples)
    AB = A @ B.T # (num_samples, num_samples)
    N = A_reduced + B_reduced - 2 * AB
    return torch.exp(- N / (N.mean() * lambda_))

def RBF_kernel(A: torch.Tensor, B: torch.Tensor, _gamma) -> torch.Tensor:
    A_reduced = (A * A).sum(dim=-1).reshape(-1, 1)  # column vector (num_samples, 1)
    B_reduced = (B * B).sum(dim=-1).reshape(1, -1)  # row vector (1, num_samples)
    AB = A @ B.T  # (num_samples, num_samples)
    N = A_reduced + B_reduced - 2 * AB
    return torch.exp(- N * _gamma)

def Poly_kernel(A: torch.Tensor, B: torch.Tensor, _) -> torch.Tensor:
    return (A @ B.T) ** 2

def sinANOVA_kernel(A: torch.Tensor, B: torch.Tensor, _) -> torch.Tensor:
    K = torch.ones((A.size(0), B.size(0)), dtype=torch.float32, device=A.device)
    for d in range(A.size(1)):
        x_d = A[:, d].unsqueeze(1)
        y_d = B[:, d].unsqueeze(0)
        sin_component = torch.sin(torch.pi * (x_d - y_d))
        K *= (1 + sin_component)
    return K

def compute_rho(X:torch.Tensor, Y:torch.Tensor,kernel, lambda_:float=1.0) -> float:
    '''
        Check if Y could be learned from X.

        X: tensor of shape (num_samples, dim_1)
        Y: tensor of shape (num_samples, dim_2)
    '''
    K = kernel(X, X, lambda_)
    l, U = ED.apply(K)
    P = U @ torch.diag(l.clamp(min=0.0, max=1.0)) @ U.T
    Y_perp = Y - P @ Y
    rho = ((Y_perp ** 2).mean(0) ** 0.5).mean()
    return rho.item()


