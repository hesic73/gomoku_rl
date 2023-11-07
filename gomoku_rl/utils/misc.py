import os
import logging
import torch
from typing import Optional,Dict
import random


class ActorBank:
    def __init__(self, dir: str, latest_n: Optional[int] = None) -> None:
        self.dir = dir
        if not os.path.isdir(self.dir):
            logging.info(f"{self.dir} doesn't exist. Create it instead.")
            os.makedirs(self.dir)
        self._cnt = 0
        self.lastest_n = latest_n
        self._index2name: Dict[int, str] = {}

    def __len__(self) -> int:
        return len(self._index2name)

    def save(self, policy, name: Optional[str] = None):
        name = name or f"actor_{self._cnt}.pt"
        path = os.path.join(self.dir, name)
        torch.save(policy, path)
        self._index2name.update({self._cnt: name})
        self._cnt += 1

        if self.lastest_n is not None and self._cnt > self.lastest_n:
            os.remove(os.path.join(self.dir,self._index2name[self._cnt - self.lastest_n - 1]))
            del self._index2name[self._cnt - self.lastest_n - 1]
            
        return path

    def get_actor(self, name: str):
        path = os.path.join(self.dir, name)
        ckpt = torch.load(path)
        return ckpt

    def get_actor_by_index(self, index: int):
        name = self._index2name[index]
        return self.get_actor(name)

    def get_latest(self):
        assert self._cnt > 0
        return self.get_actor_by_index(self._cnt - 1)

    def get_actor_paths(self):
        return [os.path.join(self.dir,v) for v in self._index2name.values()]
    
    def get_random(self):
        path=random.choice(self.get_actor_paths())
        ckpt = torch.load(path)
        return ckpt