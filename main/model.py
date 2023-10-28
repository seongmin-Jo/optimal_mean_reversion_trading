import ou_mle as ou
import numpy as np
import pandas as pd
from OptimalStopping import OptimalStopping
from datetime import datetime
from collections import deque


class Model:
    '''
    How to use Model:
    1. Model.Update() in OnData (including during Warmup)
    2. if Model.Ready2Train() -> Model.Train()
        2.1. Retrain periodically
    3. Buy Portfolio if Model.IsEnter()
    4. If bought, sell if Model.IsExit()
    '''

    def __init__(self):
        self.optimal_stopping = None
        self.alloc_B = -1

        self.time = deque(maxlen=252)  # RW's aren't supported for datetimes
        self.close_A = deque(maxlen=252)
        self.close_B = deque(maxlen=252)

        self.portfolio = None  # represents portfolio value of holding 
        # $1 of stock A and -$alloc_B of stock B

    def Update(self, time, close_A, close_B):
        '''
        Adds a new point of data to our model, which will be used in the future for training/retraining
        '''
        if self.portfolio is not None:
            self.portfolio.Update(close_A, close_B)

        self.time.append(time)
        self.close_A.append(close_A)
        self.close_B.append(close_B)

    # @property basically a function to a field
    @property
    def Ready2Train(self):
        '''
        returns true iff our model has enough data to train
        '''
        return len(self.close_A) == self.close_A.maxlen

    @property
    def IsReady(self):
        '''
        returns true iff our model is ready to provide signals
        '''
        return self.optimal_stopping is not None

    def Train(self, r=.05, c=.05):
        '''
        Computes our OU and B-Allocation coefficients
        '''
        if not self.Ready2Train:
            return

        ts_A = np.array(self.close_A)
        ts_B = np.array(self.close_B)

        days = (self.time[-1] - self.time[0]).days
        dt = 1.0 / days

        theta, mu, sigma, self.alloc_B = self.__argmax_B_alloc(ts_A, ts_B, dt)

        try:
            if self.optimal_stopping is None:
                self.optimal_stopping = OptimalStopping(theta, mu, sigma, r, c)
            else:
                self.optimal_stopping.UpdateFields(theta, mu, sigma, r, c)
        except:
            # sometimes we get weird OU Coefficients that lead to unsolveable Optimal Stopping
            self.optimal_stopping = None

        self.portfolio = Portfolio(ts_A[-1], ts_B[-1], self.alloc_B)

    def AllocationB(self):
        return self.alloc_B

    def IsEnter(self):
        '''
        Return True if it is optimal to enter the Pairs Trade, False otherwise
        '''
        return self.portfolio.Value() <= self.optimal_stopping.Entry()

    def IsExit(self):
        '''
        Return True if it is optimal to exit the Pairs Trade, False otherwise
        '''
        return self.portfolio.Value() >= self.optimal_stopping.Exit()

    def __compute_portfolio_values(self, ts_A, ts_B, alloc_B):
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

    def __argmax_B_alloc(self, ts_A, ts_B, dt):
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
            portfolio_values = self.__compute_portfolio_values(ts_A, ts_B, x)
            return ou.estimate_coefficients_MLE(portfolio_values, dt)

        vectorized = np.vectorize(compute_coefficients)
        linspace = np.linspace(.01, 1, 100)
        res = vectorized(linspace)
        index = res[3].argmax()

        return res[0][index], res[1][index], res[2][index], linspace[index]

    def get_coefficients(self):
        '''
        Returns the OU coefficients of our model
        '''
        if not self.IsReady:
            return None
        return self.optimal_stopping.theta, self.optimal_stopping.mu, self.optimal_stopping.sigma

    def __repr__(self):
        '''
        String representation of the OU coefficients of our model
        '''
        return f'θ: {self.optimal_stopping.theta:.2} μ: {self.optimal_stopping.mu:.2} σ: {self.optimal_stopping.sigma:.2}' \
            if self.IsReady else 'Not ready'


class Portfolio:
    '''
    Represents a portfolio of holding $1 of stock A and -$alloc_B of stock B
    '''

    def __init__(self, price_A, price_B, alloc_B):
        self.init_price_A = price_A
        self.init_price_B = price_B
        self.curr_price_A = price_A
        self.curr_price_B = price_B
        self.alloc_B = alloc_B

    def Update(self, new_price_A, new_price_B):
        self.curr_price_A = new_price_A
        self.curr_price_B = new_price_B

    def Value(self):
        return self.curr_price_A / self.init_price_A - self.alloc_B * self.curr_price_B / self.init_price_B