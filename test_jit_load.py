import torch
from gomoku_rl import GomokuEnv

m = torch.jit.load("tsmodule.pt")
print(isinstance(m,torch.jit.ScriptModule))
env=GomokuEnv(num_envs=16,board_size=15)
td=env.reset()[[0]]
with torch.no_grad():
    td=m(td["observation"], td["action_mask"])
    
print(td)