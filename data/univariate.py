from .basicdata import BasicData
import numpy as np
from typing import TypeVar, Union


class SingleV(BasicData):
    T = TypeVar('T')

    @classmethod
    def create(cls, data: Union[np.array, list]) -> T:
        return cls(data)

