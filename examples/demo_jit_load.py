import torch
from gomoku_rl import GomokuEnv

m: torch.jit.ScriptModule = torch.jit.load("tsmodule.pt")
env = GomokuEnv(num_envs=16, board_size=15)
td = env.reset()[[0]]
with torch.no_grad():
    td = m(td["observation"], td["action_mask"])

print(td)
