import numpy as np

def euclidean_distance(x1 : np.ndarray , x2 : np.ndarray):
    
    # d = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 )

    d = np.sqrt(np.sum((x1-x2)**2))

    return d
    