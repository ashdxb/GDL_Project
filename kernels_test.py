import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import rho

color = 'black'
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 10
plt.rcParams['font.size'] = 30
plt.rcParams['xtick.color'] = color
plt.rcParams['ytick.color'] = color
# labels
plt.rcParams['axes.labelcolor'] = color
plt.rcParams['axes.labelsize'] = 20
# grid
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = color
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.grid'] = True

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from rho import *
del sys.path[0]


def test_1(kernel=rho.Gau_kernel):
    columns = ['Recover X from Z:', 'Recover X from W:', 'Recover Z from X:', 'Recover W from X:']

    data = []
    for _ in tqdm(range(100)):
        num_samples = 1000
        dim = 1
        X = torch.tensor(np.random.normal(size=(num_samples, dim)).astype(np.float32))  # normally distributed r.v.
        Z = torch.sign(X) * (X ** 2)  # full information
        W = X ** 2  # partial information

        rho_list = []
        rho_list.append(compute_rho(Z, X,kernel))  # Recover X from Z
        rho_list.append(compute_rho(W, X,kernel))  # Recover X from W
        rho_list.append(compute_rho(X, Z,kernel))  # Recover Z from X
        rho_list.append(compute_rho(X, W,kernel))  # Recover W from X
        data.append(rho_list)
    df = pd.DataFrame(data=np.array(data), columns=columns)

    df2 = pd.DataFrame()
    df2['test'] = columns
    df2['mean'] = df.mean().values
    df2['std'] = df.std().values
    df2['theoretical value'] = [0, 1, 0, 0]
    print()
    print(df2)

def test_2(kernel, ax, graph):
    dim = 1
    torch.manual_seed(4)

    dt_df = pd.DataFrame()
    steps = np.linspace(-1, 1, 200).astype(np.float32)
    dt_df['steps'] = steps
    X = torch.tensor(np.random.normal(size=(1000, dim)).astype(np.float32))
    N = torch.tensor(np.random.normal(size=(1000, dim)).astype(np.float32))

    rho_list = []
    for alpha in tqdm(steps):
        Y = (X + alpha * N).sum(dim=1, keepdim=True)
        rho_list.append(compute_rho(X, Y, kernel, lambda_=1.0))
    dt_df['Theoretical'] = np.abs(steps) * np.sqrt(dim)
    dt_df['Estimated'] = np.array(rho_list).astype(np.float32)
    if graph == "Theoretical scaled Estimated":
        dt_df['Estimated'] = dt_df['Estimated'] / dt_df['Estimated'].max() * dt_df['Theoretical'].max()
    elif graph == "Estimated only":
        dt_df['Theoretical'] = dt_df['Estimated']


    x = dt_df['steps'].values
    ax.plot(x, dt_df['Theoretical'].values, linewidth=5, label='Theoretical')
    ax.plot(x, dt_df['Estimated'].values, linewidth=5, label='Estimated')
    ax.set_xlabel(r'$\alpha$', fontsize=15)
    ax.set_ylabel(r'$\rho^*(Y|X)$', fontsize=15)
    ax.set_title(kernel.__name__)
    return ax
    # plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'rho_test', 'test_2.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test rho')
    parser.add_argument('--test', default=1, type=int, help='Run test 1 or 2.')
    args = parser.parse_args()


    np.random.seed(4)
    torch.manual_seed(4)
    for graph in ["Theoretical", "Theoretical scaled Estimated", "Estimated only"]:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        axs = axs.flatten()
        for kernel, ax in zip([rho.Gau_kernel, rho.RBF_kernel, rho.Poly_kernel, rho.sinANOVA_kernel], axs):
            print(f"Testing {kernel.__name__}")
            if args.test != 1:
                test_1(kernel)
            else:
                test_2(kernel, ax, graph)
        plt.tight_layout()
        # plt.legend()
        plt.show()
