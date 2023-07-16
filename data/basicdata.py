import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar, Union
import rich


class BasicData(object):
    T = TypeVar('T')
    def __init__(self, data = None):
        super().__init__()
        self.data = np.array(data)

    @classmethod
    def create_basic_data(cls, data: Union[np.ndarray, list]) -> T:
        se = cls()
        se.data = np.array(data)
        return se

    def show(self):
        rich.print(self.data)

    @classmethod
    def plot(cls, x: T, curves: list[T], form: str= '-') -> None:
        fig, ax = plt.subplots()
        for curve in curves:
            # print(type(curve))
            ax.plot(x.data, curve.data, form)
        plt.show()