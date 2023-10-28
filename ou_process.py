import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Ornstein-Uhlenbeck process is not just stationary but also normally distributed
# ou parameter : dXt = mu(theta - Xt)dt + sigma * dBt
#      - Bt : standard Brownian motion under the probability measure P
#      - mu : deterministic part & the drift of the process, mean-reversion rate
#      - sigma : control the random process 
#      - θ : long-term mean, θ ∈ R
#      - If sigma is large enough, then mu become unsignificant for the process


# Source of Mathmatical Calculation
# https://www.ubs.com/global/en/investment-bank/in-focus/research-focus/quant-answers/quant-insight-series/_jcr_content/mainpar/toplevelgrid_7262680_322968126/col1/actionbutton_3358030.1305440028.file/PS9jb250ZW50L2RhbS9pbnRlcm5ldGhvc3RpbmcvaW52ZXN0bWVudGJhbmsvZW4vZXF1aXRpZXMvcWlzLXZpcnR1YWwtZXZlbnQtZGVja3MtMjAyMC9tci10cmFkaW5nLXRhbGstdWJzLWFwcmlsLTIwMjEucGRm/mr-trading-talk-ubs-april-2021.pdf
# https://github.com/mghadam/ouparams/blob/main/src/ouparams/ouparams.py
# https://reference.wolfram.com/language/ref/OrnsteinUhlenbeckProcess.html

def get_optimal_ou_params(X, dt): 
    """
    Xx = Σ(i=1, i=n) x(i-1)
    Xy = Σ(i=1, i=n) x(i)
    Xxx = Σ(i=1, i=n) x(i-1)^2
    Xyy = Σ(i=1, i=n) x(i)^2
    Xxy = Σ(i=1, i=n) x(i) * x(i-1)
    
    something uncorrect for finding optimal mu, so we get mu by another way
    """
    n = X.size
    Xx  = np.sum(X[:-1])
    Xy  = np.sum(X[1:])
    Xxx = np.sum(X[:-1]**2)
    Xyy = np.sum(X[1:]**2)
    Xxy = np.sum(X[:-1] * X[1:])
        
    theta_star = ((Xy * Xxx) - (Xx * Xxy))/(n*(Xxx - Xxy) - ((Xx**2) - Xx * Xy)) # Mean
    mu_star = -(1 / dt) * np.log((Xxy - theta_star * Xx - theta_star * Xy + n*theta_star**2)/(Xxx - 2 * theta_star * Xx + n * theta_star**2))    
    alpha = np.exp(-mu_star * dt) # Rate
    sigma_h = np.sqrt((1/n)*(Xyy-(2*alpha*Xxy)+((alpha**2)*Xxx)-(2*theta_star*(1-alpha)*(Xy-alpha*Xx))+(n*(theta_star**2)*(1-alpha)**2)))
    sigma_star = np.sqrt((sigma_h**2)*(2*mu_star/(1-alpha**2)))  # Volatility

    return theta_star, mu_star, sigma_star


"""Inaccurate Result
def fit_ar1(ts: pd.Series, dt:float = 1/252) -> (np.array, float):
    
    # Fit AR(1) process from time series of price
    
    ts_y = ts.values[1:].reshape(-1, 1)
    ts_x = np.append(np.ones(len(ts_y)), ts.values[:-1]).reshape(2,-1).T
    
    phi = np.linalg.inv(ts_x.T @ ts_x) @ ts_x.T @ ts_y
    sigma = np.sqrt(np.sum((ts_y - ts_x @ phi) ** 2) / (len(ts_y)))
    phi = phi.reshape(-1)
    
    theta = phi[0] / (1-phi[1])
    mu = (1-phi[1]) / dt
    sigma = sigma / np.sqrt(dt)
    return theta, mu, sigma
"""

def compose_xt(s1, s2, A, B):
    alpha = A / s1[0]
    beta = B / s2[0]
    return alpha * s1 - beta * s2


# Define the OU probability density function
def f_OU(xi, xi_1, theta, mu, sigma, dt):
    sigma_tilde_squared = sigma**2 * (1 - np.exp(-2 * mu * dt)) / (2 * mu)
    exponent = -(xi - xi_1 * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt)))**2 / (2 * sigma_tilde_squared)
    return (1 / np.sqrt(2 * np.pi * sigma_tilde_squared)) * np.exp(exponent)


# Define the average log likelihood
def avg_log_likelihood(params, Xt, dt):
    theta, mu, sigma = params
    n = len(Xt)
    sigma_tilde = np.sqrt(sigma**2 * (1 - np.exp(-2 * mu * dt)) / (2 * mu))
    
    sum_term = sum([(Xt[i] - Xt[i-1] * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt)))**2 for i in range(1, n)])
    
    log_likelihood = (-0.5 * np.log(2 * np.pi) - np.log(sigma_tilde) - (1 / (2 * n * sigma_tilde**2)) * sum_term)
    return -log_likelihood

# Same Result
def get_average_log_likelihood(theta, mu, sigma, X, dt):
    sigma_square = sigma**2 * (1 - np.exp(-2*mu*dt)) / (2 * mu)
    sigma_tilda = np.sqrt( sigma_square )
    
    N = X.size
    
    term1 = -0.5 * np.log(2 * np.pi)
    term2 = -np.log(sigma_tilda)
    
    prefactor = -1 / (2 * N * sigma_tilda**2)
    sum_term = 0
    for i in range(1, N):
        x2 = X[i]
        x1 = X[i-1]
        
        sum_term = sum_term + (x2 - x1 * np.exp(-mu*dt) - \
                   theta * (1-np.exp(-mu*dt)))**2
    
    f = (term1 + term2 + prefactor * sum_term)
    return f

def adjust_mu_maximizing_avg_log_likelihood(xt, theta, sigma):
    adjusted_initial_params = [theta, 0.1, sigma]
    bounds = [(-np.inf, np.inf), (0, np.inf), (0, np.inf)]
    # Minimize the negative log likelihood with adjusted initial guesses
    result_adjusted = minimize(avg_log_likelihood, adjusted_initial_params, args=(xt, dt), bounds=bounds, method='L-BFGS-B')

    # Extract the optimized parameters
    theta_opt_adj, mu_opt_adj, sigma_opt_adj = result_adjusted.x

    # Calculate the average log likelihood with optimized parameters
    avg_ll_opt_adj = -avg_log_likelihood([theta_opt_adj, mu_opt_adj, sigma_opt_adj], xt, dt)
    return theta_opt_adj, mu_opt_adj, sigma_opt_adj, avg_ll_opt_adj


def get_mle_table(s1, s2, dt):

    candidate_B = np.linspace(0.001, 1, 1000)
    optimal_params_table = []
    for B in candidate_B:
        X = compose_xt(s1, s2, 1, B)
        theta_star, mu_star, sigma_star = get_optimal_ou_params(X, dt)
        mle = get_average_log_likelihood(theta_star, mu_star, sigma_star, X, dt)
        optimal_params_table.append((theta_star, mu_star, sigma_star, mle, B))

    table = pd.DataFrame(optimal_params_table, columns=['theta', 'mu', 'sigma', 'mle', 'B'])
    return table


def get_B_star(table):
    return float(table.iloc[np.where(table.mle==np.max(table.mle))].B.values)


"""
if __name__ == "__main__":
    # Same example 
    GDX = pd.read_csv('data/GDX_historical.csv')
    GLD = pd.read_csv('data/GLD_historical.csv')
    SLV = pd.read_csv('data/SLV_historical.csv')

    gld = GLD['Adj Close'].to_numpy()
    gdx = GDX['Close'].to_numpy()
    slv = SLV['Close'].to_numpy()

    M = 3
    gld = gld[M:M + 200]
    gdx = gdx[M:M + 200]
    slv = slv[M:M + 200]

    N = gld.size
    dt = 1/252

    table1 = get_mle_table(gld, gdx, dt)
    table2 = get_mle_table(gld, slv, dt)

    B = np.linspace(0.001, 1, 1000)
    plt.plot(B, table1.mle, 'k', label = 'GLD-GDX')
    plt.plot(B, table2.mle, ':', label = 'GLD-SLV')
    plt.xlabel('B')
    plt.ylabel('Log Likelihood')
    plt.grid()
    plt.legend()

    B_star_gld_gdx = get_B_star(table1)
    B_star_gld_slv = get_B_star(table2)

    print('GLD-GDX: ', B_star_gld_gdx )
    print('GLD-SLV: ', B_star_gld_slv )
    
    xt = compose_xt(gld, gdx, 1, B_star_gld_gdx)
    x2t = compose_xt(gld, slv, 1, B_star_gld_slv)
    
    theta, mu, sigma = get_optimal_ou_params(xt, dt=1/252)
    theta2, mu2, sigma2 = get_optimal_ou_params(x2t, dt=1/252)
    
    theta_star, mu_star, sigma_star, mle = adjust_mu_maximizing_avg_log_likelihood(xt, theta, sigma)
    theta_star2, mu_star2, sigma_star2, mle2 = adjust_mu_maximizing_avg_log_likelihood(x2t, theta2, sigma2)
    
    print('theta_star, mu_star, sigma_star, mle: ', theta_star, mu_star, sigma_star, mle)
    print('theta_star, mu_star, sigma_star, mle: ', theta_star2, mu_star2, sigma_star2, mle2) 
"""


import math
from math import sqrt, exp, log  # exp(n) == e^n, log(n) == ln(n)
import scipy.optimize as so
import numpy as np

def __compute_log_likelihood(params, *args):
    '''
    Compute the average Log Likelihood, this function will by minimized by scipy.
    Find in (2.2) in linked paper

    returns: the average log likelihood from given parameters
    '''
    # functions passed into scipy's minimize() needs accept one parameter, a tuple of
    #   of values that we adjust to minimize the value we return.
    #   optionally, *args can be passed, which are values we don't change, but still want
    #   to use in our function (e.g. the measured heights in our sample or the value Pi)

    theta, mu, sigma = params
    X, dt = args
    n = len(X)

    sigma_tilde_squared = sigma ** 2 * (1 - exp(-2 * mu * dt)) / (2 * mu)
    summation_term = 0

    for i in range(1, len(X)):
        summation_term += (X[i] - X[i - 1] * exp(-mu * dt) - theta * (1 - exp(-mu * dt))) ** 2

    summation_term = -summation_term / (2 * n * sigma_tilde_squared)

    log_likelihood = (-log(2 * math.pi) / 2) + (-log(sqrt(sigma_tilde_squared))) + summation_term

    return -log_likelihood
    # since we want to maximize this total log likelihood, we need to minimize the
    #   negation of the this value (scipy doesn't support maximize)


def estimate_coefficients_MLE(X, dt, tol=1e-4):
    '''
    Estimates Ornstein-Uhlenbeck coefficients (θ, µ, σ) of the given array
    using the Maximum Likelihood Estimation method

    input: X - array-like time series data to be fit as an OU process
           dt - time increment (1 / days(start date - end date))
           tol - tolerance for determination (smaller tolerance means higher precision)
    returns: θ, µ, σ, Average Log Likelihood
    '''

    bounds = ((None, None), (1e-5, None), (1e-5, None))  # theta ∈ ℝ, mu > 0, sigma > 0
                                                           # we need 1e-10 b/c scipy bounds are inclusive of 0, 
                                                           # and sigma = 0 causes division by 0 error
    theta_init = np.mean(X)
    initial_guess = (theta_init, 100, 100)  # initial guesses for theta, mu, sigma
    result = so.minimize(__compute_log_likelihood, initial_guess, args=(X, dt), bounds=bounds)
    theta, mu, sigma = result.x 
    max_log_likelihood = -result.fun  # undo negation from __compute_log_likelihood
    # .x gets the optimized parameters, .fun gets the optimized value
    return theta, mu, sigma, max_log_likelihood

def compute_portfolio_values(ts_A, ts_B, alloc_B):
    '''
    Compute the portfolio values over time when holding $1 of stock A 
    and -$alloc_B of stock B
    
    input: ts_A - time-series of price data of stock A,
           ts_B - time-series of price data of stock B
    outputs: Portfolio values of holding $1 of stock A and -$alloc_B of stock B
    '''
    
    ts_A = ts_A.copy()  # defensive programming
    ts_B = ts_B.copy()
    
    ts_A = ts_A / ts_A[0]
    ts_B = ts_B / ts_B[0]
    return ts_A - alloc_B * ts_B

def arg_max_B_alloc(ts_A, ts_B, dt):
    '''
    Finds the $ allocation ratio to stock B to maximize the log likelihood
    from the fit of portfolio values to an OU process

    input: ts_A - time-series of price data of stock A,
           ts_B - time-series of price data of stock B
           dt - time increment (1 / days(start date - end date))
    returns: θ*, µ*, σ*, B*
    '''
    
    theta = mu = sigma = alloc_B = 0
    max_log_likelihood = 0

    def compute_coefficients(x):
        portfolio_values = compute_portfolio_values(ts_A, ts_B, x)
        return estimate_coefficients_MLE(portfolio_values, dt)
    
    vectorized = np.vectorize(compute_coefficients)
    linspace = np.linspace(.01, 1, 100)
    res = vectorized(linspace)
    index = res[3].argmax()
    
    return res[0][index], res[1][index], res[2][index], linspace[index]