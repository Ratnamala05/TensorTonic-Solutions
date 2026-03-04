import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    """
    Compute sinusoidal positional encoding matrix.
    """
    
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    
    angle_rates = 1 / (base ** (2 * (i//2) / d_model))
    angles = pos * angle_rates
    
    PE = np.zeros((seq_len, d_model))
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    
    return PE