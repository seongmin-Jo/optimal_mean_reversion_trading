import numpy as np
from scipy.integrate import quad
from typing import Callable

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
    
    def F_prime(x, u):
        term1 = u**(r/mu)
        term2 = np.sqrt(2*mu/(sigma**2))
        term3 = np.exp( term2 * (x-theta) * u - (u**2)/2 )
        return term2 * quad(term1 * term3,  0, np.inf, args=(x,))[0]
    
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
    
    def G_prime(x, u):
        term1 = u**(r/mu)
        term2 = np.sqrt(2*mu/(sigma**2))
        term3 = np.exp( term2 * (theta-x) * u - (u**2)/2 )
        return term2 * quad(term1 * term3,  0, np.inf, args=(x,))[0]
    
    return G_prime


def expected_discounted_factor(mu: np.float32,
        sigma: np.float32,
        theta: np.float32,
        r: np.float32,
        x, 
        kappa):
    """
    E[e^(-r*tauk)]
    """

    F = function_F(mu=mu, sigma=sigma, theta=theta, r=r)
    G = function_G(mu=mu, sigma=sigma, theta=theta, r=r)

    if x > kappa or x == kappa:
        return G(x)/G(kappa)

    elif x < kappa:
        return F(x)/F(kappa)


def psi(mu: np.float32,
        sigma: np.float32,
        theta: np.float32,
        r: np.float32,
        x):

    F = function_F(mu=mu, sigma=sigma, theta=theta, r=r)
    G = function_G(mu=mu, sigma=sigma, theta=theta, r=r)
    return F(x)/G(x)