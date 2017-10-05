import pandas
import numpy as np
from scipy.optimize import curve_fit
'''
    Input: Calculate lifetime error from channel, slope, and intercept + errors
'''
def calc_lifetime_error(m, dm, c, dc, b, db):
    mc_error = np.sqrt((dm/m)**2 + (dc/c)**2)*m*c

    return np.sqrt(mc_error**2 + db**2)

'''Equation for exponential fit'''
def gaussian_fit(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

'''Equation for exponential fit'''
def exp_fit(x, a, b, c):
    return a*np.exp(-b*x) + c
'''
    Input: Residuals, xdata, ydata
    Output: R-squared value (should be between 0 and 1
'''
def calc_rsquared(residuals, xdata, ydata):
    ss_res = residuals.apply(lambda res: res**2).sum()
    ss_tot = (ydata-ydata.mean()).apply(lambda diff: diff**2).sum()

    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

'''
    Input: Dataframe with Channel and Counts, MIN_RANGE, MAX_RANGE
    Output[0]: Dataframe with Fit and Residuals added
    Output[1]: Fit parameters
    Output[2]: Fit covariance
'''
def fit_exponent(data, MIN_RANGE, MAX_RANGE):
    actual_data = data.loc[(data['Channel'] >= MIN_RANGE) & (data['Channel'] <= MAX_RANGE), ['Channel', 'Counts']]
    popt, pcov = curve_fit(exp_fit, actual_data['Channel'], actual_data['Counts'], p0=[1e4, 1e-3, 1])

    actual_data['Fit'] = actual_data['Channel'].apply(lambda chan: exp_fit(chan, popt[0], popt[1], popt[2]))
    actual_data['Residuals'] = actual_data['Counts'] - actual_data['Fit']

    return (actual_data, popt, pcov)

def fit_gaussian(x, y):
    return curve_fit(gaussian_fit, x, y, p0=[1, 0, 10])
