from typing import Protocol
from typing import Union
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt

Num = Union[float, int]

class Plot2Dable(Protocol):
    def get_x(self) -> Union[Sequence[Num], np.ndarray]:
        ...
    def get_y(self) -> Union[Sequence[Num], np.ndarray]:
        ...
    
def plot2D(points: Plot2Dable):
    fig, ax = plt.subplots()
    ax.plot(points.get_x(), points.get_y())
    plt.show()

