import numpy as np


def get_discrete_cart_pole_observation(obs: np.array) -> str:
    scaled = obs * [10, 10, 100, 10]
    discrete = np.round(scaled).astype(int)
    return ",".join(map(str, discrete))
