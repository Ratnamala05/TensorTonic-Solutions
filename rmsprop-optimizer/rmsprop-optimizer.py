import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    Returns updated parameters and accumulator.
    """
    
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)
    
    # Update squared gradient accumulator
    s = beta * s + (1 - beta) * (g ** 2)
    
    # Parameter update
    w = w - lr * g / (np.sqrt(s) + eps)
    
    return w, s