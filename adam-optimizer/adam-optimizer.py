import numpy as np

def adam_step(theta, g, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    
    theta = np.array(theta)
    g = np.array(g)
    m = np.array(m)
    v = np.array(v)

    # Update first moment
    m = beta1 * m + (1 - beta1) * g

    # Update second moment
    v = beta2 * v + (1 - beta2) * (g ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Parameter update
    theta = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)

    return theta, m, v