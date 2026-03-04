import numpy as np

def gradient_descent_quadratic(a, b, c, x_init, lr=0.01, steps=1000):
    """
    Minimize f(x) = ax^2 + bx + c using vanilla gradient descent.
    Return the final x value.
    """
    
    x = x_init
    
    for _ in range(steps):
        grad = 2*a*x + b      # derivative of ax^2 + bx + c
        x = x - lr * grad     # gradient descent update
    
    return x