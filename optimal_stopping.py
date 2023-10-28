from ou_process import *

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from typing import Callable, Tuple

import matplotlib.pyplot as plt


def function_F(mu: np.float32,
               sigma: np.float32,
               theta: np.float32,
               r: np.float32) -> Callable[np.float32, np.float32]:
    
    def f(u, x):
        term1 = u**(r/mu - 1)
        term2 = np.exp( (np.sqrt(2*mu/(sigma**2))) * (x-theta) * u - (u**2)/2 )
        return term1 * term2
    
    def F(x):
        return quad(f, 0, np.inf, args=(x,))[0]

    return F
    

def function_F_prime(mu: np.float32,
                     sigma: np.float32,
                     theta: np.float32,
                     r: np.float32) -> Callable[np.float32, np.float32]:
    
    def f_prime(u, x):
        term1 = u**(r/mu)
        term2 = np.exp( np.sqrt(2*mu/(sigma**2)) * (x-theta) * u - (u**2)/2 )
        return term1 * term2
    
    
    def F_prime(x):
        term3 = np.sqrt(2*mu/(sigma**2))
        return term3 * quad(f_prime, 0, np.inf, args=(x,))[0]
    
    return F_prime
    


def function_G(mu: np.float32,
               sigma: np.float32,
               theta: np.float32,
               r: np.float32) -> Callable[np.float32, np.float32]:
    
    def g(u, x):
        term1 = u**(r/mu - 1)
        term2 = np.exp( (np.sqrt(2*mu/(sigma**2))) * (theta-x) * u - (u**2)/2 )
        return term1 * term2
    
    def G(x):
        return quad(g, 0, np.inf, args=(x,))[0]

    return G

    
    
def function_G_prime(mu: np.float32,
                     sigma: np.float32,
                     theta: np.float32,
                     r: np.float32) -> Callable[np.float32, np.float32]:
    
    def g_prime(u, x):
        term1 = u**(r/mu)
        term2 = np.exp( np.sqrt(2*mu/(sigma**2)) * (theta-x) * u - (u**2)/2 )
        return term1 * term2
    
    def G_prime(x):
        term3 = np.sqrt(2*mu/(sigma**2))
        return term3 * quad(g_prime, 0, np.inf, args=(x,))[0]
    
    return G_prime


def function_h(x, c):
    """Reward Function"""
    def h(x, c):
        return x-c
    return h


def test_expected_discounted_factor(x: np.float32, a: np.float32, b: np.float32, F, G):
    """
    test_expected_discounted_factor(a_star, b_star, a_star, F, G) # (1, 0)
    test_expected_discounted_factor(b_star, a_star, b_star, F, G) # (0, 1)
    """
    if x <= a:
        exp_a = F(x) / F(a)
        exp_b = 0  # As it's certain that tau_a < tau_b in this case
    elif x >= b:
        exp_a = 0  # As it's certain that tau_a > tau_b in this case
        exp_b = G(x) / G(b)
    else:
        exp_a = F(x) / F(a)
        exp_b = G(x) / G(b)
    return exp_a, exp_b


def expected_discounted_reward(params: Tuple[np.float32, np.float32], x: np.float32, h, F, G) -> np.float32:
    a, b = params
    
    if x <= a:
        exp_a = F(x) / F(a)
        exp_b = 0  # As it's certain that tau_a < tau_b in this case
    elif x >= b:
        exp_a = 0  # As it's certain that tau_a > tau_b in this case
        exp_b = G(x) / G(b)
    else:
        exp_a = F(x) / F(a)
        exp_b = G(x) / G(b)

    return -(h(a) * exp_a + h(b) * exp_b)


def find_optimal_interval(F,
                          G,
                          r: np.float32,
                          c: np.float32,
                          x: np.float32) -> Tuple[np.float32, np.float32]:
    
    h = lambda z: z - c

    # Optimize for a and b jointly
    initial_guess = [x - 0.05, x + 0.05]  # Replace with more educated guesses if available
    # initial_guess = [0.4690968492625811, 0.6004618825146063] # np.min(xt), np.max(xt)
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]  # Replace with actual bounds if available
    
    result = minimize(expected_discounted_reward, initial_guess, args=(x, h, F, G), bounds=bounds)
    
    candidate_a, candidate_b = result.x
    
    return candidate_a, candidate_b

def function_V(b : np.float32,
               c : np.float32, 
               F : Callable[np.float32, np.float32]) -> Callable[np.float32, np.float32]:
    
    """Input b should be b_star"""
    
    def V(x:np.float32) -> Callable[np.float32, np.float32]:
        if x < b:
            return (b-c) * F(x) / F(b)

        else:
            return x-c
    
    return V

def function_V_prime(b: np.float32,
                            c: np.float32,
                            F: Callable[np.float32, np.float32],
                            F_prime: Callable[np.float32, np.float32]) -> Callable[np.float32, np.float32]:
    
    def V_prime(x): 
        if x < b:
            return (b - c) * F_prime(x) / F(b)
        else:
            return 1

    return V_prime


def function_J(d: np.float32,
                      c: np.float32,
                      G: Callable[np.float32, np.float32],
                      V: Callable[np.float32, np.float32]) -> Callable[np.float32, np.float32]:
    
    def J(x: np.float32) -> Callable[np.float32, np.float32]:
        
        if x < d: # 0.546 < d
            return V(x) - x - c
        else: # d < 0.546 
            return (V(d) - d - c) * G(x) / G(d)
    
    return J


"""
def b_star_finder(x):
    return F(x) - (x - c) * F_prime(x)

def find_b_star(x_init):
    root_result = root_scalar(b_star_finder, bracket=[0,1], x0=x_init)
    b_star = root_result.root
    return b_star, root_result.converged

def d_star_finder(x):
    return (G_prime(x) * (V(x) - x - c)) - (G(x) * (V(x) - 1))

def find_d_star(x_init, b):
    root_result = root_scalar(d_star_finder, bracket=[0,b], x0=x_init)
    d_star = root_result.root
    return d_star, root_result.converged
"""

def find_b_star(x: np.float32,
                c: np.float32,
                F: Callable[np.float32, np.float32],
                domain: np.array) -> np.float32: 
    
    # Temporary implementation
    i_b_star = np.argmax([function_V(b=b, c=c, F=F)(x) for b in domain]) 
    b_star = domain[i_b_star]
    return b_star

    # Original implementation
    
    # def find_b_star(x, cost, F, F_prime, lr = 1/3):
    #     error = np.finfo(float).max
    #     while True:
    #         fb, fpb = F(x), F_prime(x)
    #         if fpb <=0:
    #             raise ValueError("F' is not > 0")
    #         error = (cost - x) * fpb +fb
    #         # print(f'{error} at {fb}, {fpb}, {x}')
    #         if abs(error) > 1e-14:
    #             x += lr * error/fpb
    #         else:
    #             break
    #     return x
    
def find_d_star(x: np.float32,
                c: np.float32,
                G: Callable[np.float32, np.float32],
                V: Callable[np.float32, np.float32],
                domain: np.array) -> np.float32: 
    
    # Temporary implementation
    i_d_star = np.argmax([function_J(d=d, c=c, G=G, V=V)(x) for d in domain]) 
    d_star = domain[i_d_star]
    return d_star

    # Original implementation

    # def find_d_star(x, cost, G, G_prime, V, V_prime, lr =1/100):
    #     error = np.finfo(float).max
    #     while True:
    #         g_temp, v_temp = G(x), V(x, cost) - x -cost
    #         gp, vp = G_prime(x), V_prime(x, cost) -1
    #         # if v_temp <=0:
    #         #     raise ValueError("V-b-c is not > 0")
    #         try:
    #             # error = gp/g_temp - vp/v_temp
    #             error = v_temp - g_temp
    #         except ZeroDivisionError:
    #             print(f'{x} with {error} = {v_temp} - {g_temp} and {gp}, {vp}')
    #         # print(f'{error} at {v_temp} - {g_temp}')
    #         if abs(error) > 1e-14:
    #             if vp > gp:
    #                 x += lr * error/vp
    #             else:
    #                 x -= lr * error/gp
    #         else:
    #             break
    #     return x

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
    B_star_gld_gdx = get_B_star(table1)
    B_star_gld_slv = get_B_star(table2)

    xt = compose_xt(gld, gdx, 1, B_star_gld_gdx)
    theta, mu, sigma = get_optimal_ou_params(xt, dt=1/252)

    xt = compose_xt(gld, gdx, 1, B_star_gld_gdx)
    x = xt[0]
    theta, mu, sigma = get_optimal_ou_params(xt, dt=1/252)
    r = 0.05
    c = 0.05

    F = function_F(mu=mu, sigma=sigma, theta=theta, r=r)
    F_prime = function_F_prime(mu=mu, sigma=sigma, theta=theta, r=r)
    G = function_G(mu=mu, sigma=sigma, theta=theta, r=r)
    G_prime = function_G_prime(mu=mu, sigma=sigma, theta=theta, r=r)

    candidate_a, candidate_b = find_optimal_interval(F, G, r, c, x)
    print("Optimal Interval [a, b]:", "[{}, {}]".format(candidate_a, candidate_b))

    # plot V(x) against b
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title("V(x) against b")
    plt.xlabel("b (liquidation level), d (entry level)")
    plt.ylabel("V(x) (value of liquidation)")
    plt.ylim([0, 2])
    plt.axvline(x=x, color='k', linestyle='--')
    plt.axhline(y=x, color='k', linestyle='--')

    domain = np.linspace(0.4, 1, num=100, endpoint=True)
    # domain = np.linspace(candidate_a, candidate_b, num=100, endpoint=True)

    l_V = plt.plot(domain, [function_V(b=b, c=c, F=F)(x) for b in domain], color='r', label="V(x)")

    b_star = find_b_star(x=x, c=c, F=F, domain=domain)
    plt.axvline(x=b_star, color='k', linestyle='--')
    plt.text(b_star, 0 ,f"{b_star.round(4)}")
    l_diff = plt.plot(domain,
                    [np.log(np.log(abs(F(b_star) - (b_star - 0) * F_prime(b_star)))) for b_star in domain],  # to rescale
                    color='b',
                    label="F(b) - (b - c) * F'(b)")
    plt.legend()

    # plot J(x) against d
    plt.subplot(1, 2, 2)
    plt.title("J(x) against d")
    plt.xlabel("d (entry level)")
    plt.ylabel("J(x) (value of entry)")
    plt.ylim([0, 2])
    V = function_V(b=b_star, c=c, F=F)
    V_prime = function_V(b=b_star, c=c, F=F)

    plt.plot(domain,
            [function_J(d=d, c=c, G=G, V=V)(x) for d in domain],
            color='r',
            label="J(x)")
    d_star = find_d_star(x=x, c=c, G=G, V=V, domain=domain)
    plt.axvline(x=d_star, color='k', linestyle='--')
    plt.text(d_star, 0 ,f"{d_star.round(4)}")
    plt.axvline(x=x, color='k', linestyle='--')
    plt.text(0, V(x) - x - c + 0.01 ,f"V(x) - x - c")
    plt.text(x  + 0.01, 0,f"x={x}")
    plt.plot(domain,
            [np.log(np.log(abs(G(d) * (V_prime(d) - 1) - (G_prime(d) * (V(d) - d -c))))) for d in domain], # to rescale
            color='b',
            label="G(d) * (V'(d) - 1) - (G'(d) * (V(d) - d -c)")
    plt.legend()
    plt.show()
"""

    
    
from math import sqrt, exp
import scipy.integrate as si
import scipy.optimize as so
import numpy as np

def Prime(f, x, theta, mu, sigma, r, h=1e-5):
    # given f, estimates f'(x) using the difference quotient formula 
    # WARNING: LOWER h VALUES CAN LEAD TO WEIRD RESULTS
    return (f(x+h, theta, mu, sigma, r) - f(x, theta, mu, sigma, r)) / h 

def Prime2(f, x, theta, mu, sigma, r, c, h=1e-5):
    # given f, estimates f'(x) using the difference quotient formula 
    # WARNING: LOWER h VALUES CAN LEAD TO WEIRD RESULTS
    return (f(x+h, theta, mu, sigma, r, c) - f(x, theta, mu, sigma, r, c)) / h 

def F(x, theta, mu, sigma, r):
    # equation 3.3
    def integrand(u):
        return u**(r/mu - 1) * exp(sqrt(2*mu / sigma**2) * (x-theta)*u - u**2/2)
    return si.quad(integrand, 0, np.inf)[0]

def G(x, theta, mu, sigma, r):
    # equation 3.4
    def integrand(u):
        return u**(r/mu - 1) * exp(sqrt(2*mu / sigma**2) * (theta-x)*u - u**2/2)
    return si.quad(integrand, 0, np.inf)[0]

def b_star(theta, mu, sigma, r, c):
    # estimates b* using equation 4.3
    # def opt_func(b):
    #     # equation 4.3 in the paper with terms moved to one side
    #     return abs(F(b, theta, mu, sigma, r) - (b-c)*Prime(F, b, theta, mu, sigma, r))
    # bounds = ((.01, .99),)
    # result = so.minimize(opt_func, .5, bounds=bounds)

    b_space = np.linspace(0.1,0.9, 801)
    def func(b):
        return F(b, theta, mu, sigma, r) - (b-c)*Prime(F, b, theta, mu, sigma, r)
    
    return so.brentq(func, 0, 1)

def V(x, theta, mu, sigma, r, c):
    # OUR SELL SIGNAL
    # equation 4.2, solution of equation posed by 2.3
    
    b_star_val = b_star(theta, mu, sigma, r, c)
    
    if x < b_star_val:
        return (b_star_val - c) * F(x, theta, mu, sigma, r) / F(b_star_val, theta, mu, sigma, r)
    else:
        return x - c

def d_star(theta, mu, sigma, r, c):
    # estimates d* using equation 4.11
  
    def func(d):
        return (G(d, theta, mu, sigma, r) * (Prime2(V, d, theta, mu, sigma, r, c) - 1)) - (Prime(G, d, theta, mu, sigma, r) * (V(d, theta, mu, sigma, r, c) - d - c))

    # finds the root between the interval [0, 1]
    return so.brentq(func, 0, 1)