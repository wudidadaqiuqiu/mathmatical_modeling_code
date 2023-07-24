from typing import Protocol
from typing import Union
from collections.abc import Sequence
import matplotlib.pyplot as plt

Num = Union[float, int]

class Plot2Dable(Protocol):
    def get_x(self) -> Sequence[Num]:
        ...
    def get_y(self) -> Sequence[Num]:
        ...
    
def plot2D(points: Plot2Dable):
    fig, ax = plt.subplots()
    ax.plot(points.get_x(), points.get_y())
    plt.show()

