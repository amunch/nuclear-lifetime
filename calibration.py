from __future__ import print_function

import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

from scipy.optimize import curve_fit

# To get the equation to transform channel to time, run the calibrate function.
# It will return the tuple of tuples ((y, intercept), (y error, intercept error))

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gaussian(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    mean = np.mean(x)
    std = 1

    popt, pcov = curve_fit(gauss_function, x, y, p0 = [1, mean, std])

    #plt.clf()
    #plt.scatter(x,y)

    x_gauss = np.linspace(x[0], x[-1], num=1000)
    gauss = map(lambda x: gauss_function(x, popt[0], popt[1], popt[2]), x_gauss)
    #plt.plot(x_gauss, gauss)
    #plt.show()

    return(popt[1], popt[2])

def find_gaussians():
    # Find Gaussians
    intervals = []

    df = pd.read_csv('data/time.csv')

    gaussians = []
    prev = False
    gaussian = []
    for index,row in df.iterrows():
        if row['Counts'] > 0:
            gaussian.append((row['Channel'], row['Counts']))
            prev = True
            continue
        if row['Counts'] == 0 and prev == True:
            if len(gaussian) > 1:
                gaussians.append(gaussian)
            gaussian = []
        prev = False

    gaussian_means = []
    gaussian_stds = []

    for i in gaussians:
        stats = fit_gaussian(i)
        gaussian_means.append(stats[0])
        gaussian_stds.append(stats[1])

    period = 0.32 # micro-seconds
    times = []
    for i in range(1,7): # there are 6 peaksom
        times.append(period*i)

    return(times, gaussian_means, gaussian_stds)

def calibrate():
    (times, gaussian_means, gaussian_stds) = find_gaussians()

    weights = 1/np.power(gaussian_stds, 2)

    # put x and y into a pandas DataFrame, and the weights into a Series
    ws = pd.DataFrame({
        'x': times,
        'y': gaussian_means,
        'yerr': map(lambda x: x * 1000, gaussian_stds)
    })

    wls_fit = sm.wls('x ~ y', data=ws, weights=1 / weights).fit()

    return((wls_fit.params['y'], wls_fit.params['Intercept']),(wls_fit.bse['y'], wls_fit.bse['Intercept']))
