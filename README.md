# optimal_mean_reversion_trading
Implement thesis via python : [Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit](https://arxiv.org/abs/1411.5062)

### 1. Pair Trading using ou process

#### 1.1 OU process

The [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) 
is described by the following SDE: 

$$ dX_t = \mu (\theta - X_t) dt + \sigma dB_t .$$

The parameters are:
- $\mu > 0$:  mean-reversion rate, deterministic part
- $\theta \in \mathbb{R}$:  long term mean
- $\sigma > 0$:   volatility coefficient, ontrol the random process 
- B_t : standard Brownian motion under the probability measure P

This process is Gaussian, Markovian and (unconditionally) stationary.
If sigma is large enough, then mu become unsignificant for the process


#### 1.2 Estimation of parameters from a single path




#### Reference

1. [Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit](https://arxiv.org/abs/1411.5062)

2. [Ornstein-Uhlenbeck process and applications](https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/6.1%20Ornstein-Uhlenbeck%20process%20and%20applications.ipynb)

3. [Pair Ratio Optimization](https://github.com/kpmooney/numerical_methods_youtube/blob/master/ornstein_uhlenbeck/Pair%20Ratio%20Optimization.ipynb)