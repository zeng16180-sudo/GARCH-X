# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:30:13 2026

@author: user
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/"data"
OUTPUT_DIR=BASE_DIR/"output"
OUTPUT_DIR.mkdir(exist_ok=True)

#處理x+y<value的參數
def trans_param(value,raw_total, raw_share):
    total=value*expit(raw_total)
    share_x=expit(raw_share)
    x=total*share_x
    y=total*(1-share_x)
    return x, y

def inverse_trans_param(value, x, y):
    total=x+y
    raw_total=logit(total/value)
    raw_share=logit(x/total)
    return raw_total, raw_share

def garch_x(params, r_t, x):
    mu, log_h1, raw_alpha, raw_beta, log_omega, log_gamma=params
    alpha=trans_param(1,raw_alpha, raw_beta)[0]
    beta=trans_param(1,raw_alpha, raw_beta)[1]
    h1=np.exp(log_h1)
    omega=np.exp(log_omega)
    gamma=np.exp(log_gamma)
    
    h_t_list=[h1]
    for t in range(1, len(r_t)):
        h_t=omega+alpha*(r_t[t-1]-mu)**2+beta*h_t_list[t-1]+gamma*x[t-1]
        h_t_list.append(h_t)
    z_t=(r_t-mu)/np.sqrt(h_t_list)
    
    h_t_array=np.asarray(h_t_list)
    ll=np.sum(-0.5*(np.log(h_t_array)+z_t**2))
    return ll, h_t_array, z_t

def neg_ll(params, r_t, x):
    return -garch_x(params, r_t, x)[0]


################################################################
df=pd.read_csv(DATA_DIR/"JPY_USD_data.csv", index_col=0)
df.index=pd.to_datetime(df.index,format="%Y/%m/%d")
r_t=df["r_t"].iloc[1:]
r_t=np.asarray(r_t)
x=-df["IRD"].iloc[1:]
x=np.asarray(x)

mu=0.05
h1=1e-4
alpha=0.9
beta=0.09
omega=0.001
gamma=10.0

init_params=(mu,
             np.log(h1),
             inverse_trans_param(1, alpha, beta)[0],
             inverse_trans_param(1, alpha, beta)[1],
             np.log(omega),
             np.log(gamma)
             )

result=minimize(
    neg_ll,
    init_params,
    args=(r_t, x),
    method="L-BFGS-B",
    options={"maxiter": 5000,
             "ftol": 1e-12,
             "gtol": 1e-6,}
    )

mu, log_h1, raw_alpha, raw_beta, log_omega, log_gamma=result.x

h1=np.exp(log_h1)
alpha=trans_param(1,raw_alpha, raw_beta)[0]
beta=trans_param(1,raw_alpha, raw_beta)[1]
omega=np.exp(log_omega)
gamma=np.exp(log_gamma)

params=(mu, h1, alpha, beta, omega, gamma)

params_result={"mu": mu,
               "h1": h1,
               "alpha": alpha,
               "beta": beta,
               "omega": omega,
               "gamma": gamma,
               "Log-likelihood": -result.fun}

z_t=garch_x(params, r_t, x)[2]
volatility=np.sqrt(garch_x(result.x, r_t, x)[1])

#z_t時序圖
plt.plot(df.index[1:], z_t, linewidth=0.7)
plt.title("z_t")
plt.show()

#vol時序圖
plt.plot(df.index[1:], volatility, linewidth=1.0)
plt.title("volatility")
plt.show()