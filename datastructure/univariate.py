from .basicdata import BasicData
import numpy as np
from typing import TypeVar, Union


class SingleV(BasicData):
    T = TypeVar('T')

    @classmethod
    def create(cls, data: Union[np.ndarray, list]) -> T:
        return cls(data)

