from tensordict import TensorDictBase
import torch
from torchrl.envs import EnvBase
from typing import Callable
from tqdm import tqdm

from torchrl.data import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, Sampler

from torch.cuda import _device_t


def get_replay_buffer(
    buffer_size: int,
    batch_size: int,
    sampler: Sampler | None = None,
    device: _device_t = None,
):
    storage = LazyTensorStorage(max_size=buffer_size, device=device)
    buffer = TensorDictReplayBuffer(
        storage=storage, batch_size=batch_size, sampler=sampler
    )
    return buffer



def env_next_to_agent_next(tensordict:TensorDictBase)->TensorDictBase:
    # 这里有点tricky
    # 目前没有考虑terminated，因为它是torchrl自己加的，而我肯定自己没有用到它
    # 而且似乎它和done总是一样
    next_td=tensordict['next'][:,1:]
    for key in next_td.keys():
        assert key in ("done","observation","reward","terminated")
        
    assert (tensordict['done']==tensordict['terminated']).all()
    
    td=tensordict[:,:-1]
    done=td['next']['done']
    td['next']["done"]=next_td["done"]|td['next']['done']
    td['next']["terminated"]=next_td["terminated"]|td['next']["terminated"]
    td['next']['observation']=next_td['observation']
    td['next']['reward']=td['next']['reward']-next_td['reward']*(~done).float()
    return td
    