from torchrl.data.replay_buffers.samplers import Sampler, Storage, _EMPTY_STORAGE_ERROR
import torch
from typing import Tuple, Any
import math


class SequentialSampler(Sampler):
    def __init__(self, drop_last: bool = False):
        self.len_storage = 0
        self.drop_last = drop_last
        self._ran_out = False
        self._pos: int = 0

    def _single_sample(self, len_storage: int, batch_size: int):
        index = torch.arange(self._pos, self._pos + batch_size)
        self._pos += batch_size

        # check if we have enough elements for one more batch, assuming same batch size
        # will be used each time sample is called
        if self._pos > (len_storage // batch_size - 1) * batch_size:
            self._ran_out = True
            self._pos = 0
        else:
            self._ran_out = False
        return index

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        len_storage = len(storage)
        if len_storage == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        if not len_storage:
            raise RuntimeError("An empty storage was passed")
        if self.len_storage != len_storage:
            self._pos = 0
        if len_storage < batch_size and self.drop_last:
            raise ValueError(
                f"The batch size ({batch_size}) is greater than the storage capacity ({len_storage}). "
                "This makes it impossible to return a sample without repeating indices. "
                "Consider changing the sampler class or turn the 'drop_last' argument to False."
            )
        self.len_storage = len_storage
        index = self._single_sample(len_storage, batch_size)
        # we 'always' return the indices. The 'drop_last' just instructs the
        # sampler to turn to 'ran_out = True` whenever the next sample
        # will be too short. This will be read by the replay buffer
        # as a signal for an early break of the __iter__().
        return index, {}

    @property
    def ran_out(self):
        return self._ran_out

    @ran_out.setter
    def ran_out(self, value):
        self._ran_out = value

    def _empty(self):
        self.len_storage = 0
        self._ran_out = False
        self._pos = 0
