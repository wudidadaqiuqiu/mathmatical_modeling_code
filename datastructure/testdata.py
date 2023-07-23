from .basicdata import BasicData
import numpy as np
from typing import TypeVar, Optional


class TestData(BasicData):
    T = TypeVar('T')

    @classmethod
    def create_data(cls, f, start: int, end: int, num: Optional[int]) -> tuple[BasicData, BasicData]:
        x = np.linspace(start, end, num)
        y = np.vectorize(f)(x)
        return tuple(map(BasicData.create_basic_data, [x, y]))

    @classmethod
    def create_ndarray(cls, f, start: int, end: int, num: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(start, end, num)
        y = np.vectorize(f)(x)
        return x, y
    
    