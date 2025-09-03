#Polynomial regression using gradient descent's
import numpy as np
import random
import math

#loss function
def loss_function(y_dash,y):
    return sum((p-q)**2 for p,q in zip(y_dash,y))
#signum function
def signum(x):
    return (x > 0) - (x < 0)

#Returns the value of the polynomial( having coefficients ms )at x
def polynomial(ms,x):
    sum = 0
    n = len(ms)-1
    for i , m in enumerate(ms):
        sum += m*(x**(n-i))
    return sum

#returns the loss function for predicted ms , and x values , y values being xs , ys
def polynomial_loss_function(ms,xs,ys):
    return sum((polynomial(ms,x)-y)**2 for x,y in zip(xs,ys))

#determines the gradient of the loss function with respect to the coefficient array
def coefficient_loss_gradient(ms,xs,ys):
    gradient = np.array([])
    ys_pred = polynomial(ms,xs)
    n=len(ms)
    for i in range(n):
        sum = 0
        for x,y_pred,y in zip(xs,ys_pred,ys):
            sum += 2*(x**(n-i-1))*(y_pred-y)
        gradient = np.append(gradient,sum)
    return np.array(gradient)

#determines the gradient of loss function for select number of data points

def coefficient_sample_loss_gradient(ms,sample):
    gradient = np.array([])
    xs = np.array([x for x,_ in sample])
    ys = np.array([y for _,y in sample])
    ys_pred = polynomial(ms, xs)
    n = len(ms)
    for i in range(n):
        sum = 0
        for x, y_pred, y in zip(xs, ys_pred, ys):
            sum += 2 * (x ** (n - i - 1)) * (y_pred - y)
        gradient = np.append(gradient, sum)
    return np.array(gradient)

##########POLYNOMIAL REGRESSION##########
#NOTE: Each regression function returns a tuple containing the predicted polynomial and the loss per data point for the predicted polynomial
#implements the adam optimization in polynomial regression
def adam_polyreg(data_set, n = 1, beta1 = 0.9, beta2 = 0.99, iter = 1000, learning_rate = 10**(-2), epsilon = 10**(-8),batch_size = 5):
    m_pointer = np.random.randn(n+1)*0.01
    data_set_xs = np.array([x for x, _ in data_set])
    data_set_ys = np.array([y for _, y in data_set])

    cache = np.zeros_like(m_pointer)
    velocity = np.zeros_like(m_pointer)

    for t in range(1,iter+1):
        data_sample = [data_set[random.randint(0, len(data_set) - 1)] for _ in range(batch_size)]
        grad = coefficient_sample_loss_gradient(m_pointer,data_sample)
        velocity = velocity * beta1 + (1 - beta1) * grad
        cache = cache * beta2 + (1 - beta2) * (grad ** 2)
        velocity_hat = velocity / (1 - beta1 ** t)
        cache_hat = cache / (1 - beta2 ** t)
        m_pointer = m_pointer - (learning_rate * velocity_hat) / np.sqrt(cache_hat + epsilon)
    loss_per_data_point = polynomial_loss_function(m_pointer, data_set_xs, data_set_ys) / len(data_set_xs)
    return m_pointer, loss_per_data_point

#implements the nadam optimization technique for polynomial regression
def nadam_polyreg(data_set, n = 1, beta1 = 0.9, beta2 = 0.99, iter = 1000, learning_rate = 10**(-2), epsilon = 10**(-8), batch_size = 5):
    m_pointer = np.random.randn(n + 1) * 0.01
    data_set_xs = np.array([x for x, _ in data_set])
    data_set_ys = np.array([y for _, y in data_set])

    cache = np.zeros_like(m_pointer)
    velocity = np.zeros_like(m_pointer)
    for t in range(1, iter + 1):
        data_sample = [data_set[random.randint(0, len(data_set) - 1)] for _ in range(batch_size)]
        grad = coefficient_sample_loss_gradient(m_pointer, data_sample)
        velocity = velocity * beta1 + (1 - beta1) * grad
        cache = cache * beta2 + (1 - beta2) * (grad ** 2)
        velocity_hat = velocity / (1 - beta1 ** t)
        cache_hat = cache / (1 - beta2 ** t)
        look_ahead = beta1 * velocity_hat + (1 - beta1) * grad
        m_pointer = m_pointer - (learning_rate *  look_ahead) / np.sqrt(cache_hat + epsilon)
    loss_per_data_point = polynomial_loss_function(m_pointer,data_set_xs,data_set_ys)/len(data_set_xs)
    return m_pointer,loss_per_data_point

