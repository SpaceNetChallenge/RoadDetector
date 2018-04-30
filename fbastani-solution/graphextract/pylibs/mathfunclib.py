#
# Mathematical function library.
# Author: James P. Biagioni (jbiagi1@uic.edu)
# Company: University of Illinois at Chicago
# Created: 12/16/10
#

import math

# Normal distribution PDF. Formula obtained from: http://en.wikipedia.org/wiki/Normal_Distribution
def normal_distribution_pdf(x, mu, sigma, numerator=1.0):
    return (numerator / math.sqrt(2.0 * math.pi * math.pow(sigma, 2.0))) * math.exp(-1.0 * (math.pow((x - mu), 2.0) / (2.0 * math.pow(sigma, 2.0))))

# Normal distribution CDF. Formula obtained from: http://en.wikipedia.org/wiki/Normal_Distribution
def normal_distribution_cdf(x, mu, sigma):
    return (0.5 * (1.0 + erf( (x - mu) / math.sqrt(2.0 * math.pow(sigma, 2.0)))))

# Complementary normal distribution CDF. Formula obtained from: http://en.wikipedia.org/wiki/Cumulative_distribution_function
def complementary_normal_distribution_cdf(x, mu, sigma):
    return (1.0 - normal_distribution_cdf(x, mu, sigma))

# Spring force. Formula obtained from: http://en.wikipedia.org/wiki/Hooke%27s_law
def spring_force(x, k):
    return ((-1.0 * k) * x)

# Gaussian error function. Algorithm obtained from: http://www.johndcook.com/python_erf.html
def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)
    
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    
    return sign*y
