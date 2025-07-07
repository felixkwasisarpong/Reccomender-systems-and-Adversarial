# dp_utils.py
import numpy as np
import math

def compute_rdp(q, noise_multiplier, steps, orders):
    rdp = []
    for alpha in orders:
        if q == 0:
            rdp.append(0)
        elif noise_multiplier == 0:
            rdp.append(np.inf)
        else:
            term1 = alpha / (2 * noise_multiplier**2)
            rdp.append(term1 * q**2)
    return np.array(rdp) * steps

def get_privacy_spent(orders, rdp, delta):
    eps = rdp - math.log(delta) / (np.array(orders) - 1)
    idx = np.argmin(eps)
    return eps[idx], orders[idx]