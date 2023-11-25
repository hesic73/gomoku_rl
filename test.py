import numpy as np
from gomoku_rl.utils.psro import solve_nash

payoffs = np.array(
    [
        [-0.17382812, -0.19921875, 0.99804688],
        [-0.25195312, -0.17773438, 0.98828125],
        [0.9921875, 0.99804688, 0.99609375],
    ]
)


meta_policy = solve_nash(payoffs)
print(meta_policy)
