from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union, Optional
from queue import SimpleQueue
import numpy as np
from ..show.graph import draw_digraph, DiGraphable
from ..show.plot import plot2D

@dataclass
class Point:
    data: tuple

    def __str__(self) -> str:
        return str(self.data)
    
    def __repr__(self) -> str:
        return str(self)

@dataclass
class ProbabilityMass:
    # point: Point
    probability: float

    def __str__(self) -> str:
        return str(self.probability)
    
    def __repr__(self) -> str:
        return str(self)

point = tuple
probability = float

class ProbabilityMassFunction(object):
    def __init__(self, pmf: Union[list[tuple[point, probability]], dict[point, probability], None], name: str='') -> None:
        super().__init__()
        self._data: dict[point, probability] = {}
        self.__sum_of_probability: float = 0
        self.name: str = name
        self._expectation: float
        if not pmf:
            self.dimension = 0
            return
        self.dimension = len(pmf[0][0]) if isinstance(pmf, list) else len(list(pmf.keys())[0])
        pmf_: Iterable = pmf if isinstance(pmf, list) else pmf.items()
        for pm in pmf_:
            if len(pm[0]) != self.dimension:
                raise ValueError(f'{self.dimension}, {pm[0]}')
            self._data[pm[0]] = pm[1]
            self.__sum_of_probability += pm[1]
        if not self.is_sum_one():
            raise ValueError(f'概率和不为1, {self.__sum_of_probability}')
    
    def __str__(self) -> str:
        res = f'{self.name}\n' if self.name else ''
        res += 'point\t| probability\n'
        for item in self._data.items():
            res += f'{([num for num in item[0]])}\t| {item[1]}\n'
        return res

    def __repr__(self) -> str:
        return str(self)
    
    def is_sum_one(self) -> bool:
            self.__sum_of_probability = round(sum([a for a in self._data.values()]), 12)
            return self.__sum_of_probability == 1

    def distribution_law(self) -> list:
        return sorted(self._data.items(), key=lambda x:x[0])

    def distribution(self) -> list:
        if self.dimension != 1:
            return sorted(self._data.keys())
        return [x[0] for x in sorted(self._data.keys())]

    def expectation(self) -> float:
        if hasattr(self, '_expectation'):
            return self._expectation
        assert self.dimension == 1
        self._expectation = sum([point[0] * probability for point, probability in self._data.items()])
        return self._expectation
    
    def plot(self):
        plot_pmf(self)

PMF = ProbabilityMassFunction
def merge_two_pmf(pmf1: PMF, pmf2: PMF, ndigits: int=5) -> PMF:
    assert pmf1.dimension == pmf2.dimension == 1
    res_pmf = PMF(None, name = pmf1.name + '+' + pmf2.name)
    res_pmf.dimension = 1
    pmf_dict = res_pmf._data
    for x1 in pmf1.distribution_law():
        for x2 in pmf2.distribution_law():
            key = (round(x1[0][0] + x2[0][0], ndigits),)
            pmf_dict[key] = pmf_dict.get(key, 0) + x1[1] * x2[1]
    return res_pmf

class PMFTree(object):
    def __init__(self, father: Optional[PMFTree]) -> None:
        super().__init__()
        self.father: Optional[PMFTree] = father
        self.lchild: PMFTree
        self.rchild: PMFTree
        self.data: PMF
    
    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return str(self)
        
    def __hash__(self) -> int:
        return hash((self.data))
    
    def __eq__(self, __value: PMFTree) -> bool:
        return self.data == __value.data
    
    def graph(self):
        graph_pmftree(self)

def create_pmftree(pmfs: list[PMF], father: Optional[PMFTree]=None) -> PMFTree:
    """生成的树的叶子节点有data,没有child属性"""
    tree, lenth = PMFTree(father), len(pmfs)
    if lenth == 1:
        tree.data = pmfs[0]
        return tree
    tree.lchild, tree.rchild = create_pmftree(pmfs[:lenth//2], tree), create_pmftree(pmfs[lenth//2:], tree)
    tree.data = merge_two_pmf(tree.lchild.data, tree.rchild.data)
    return tree

@dataclass
class _DiGraphablePMFTree:
    root: PMFTree

    def get_nodes(self) -> list[PMFTree]:
        res, i, lenth = [self.root], 0, 1
        while i < lenth:
            if hasattr(res[i], 'lchild'):
                res.append(res[i].lchild)
                lenth +=1
            if hasattr(res[i], 'rchild'):
                res.append(res[i].rchild)
                lenth +=1
            i += 1
        return res
    
    def get_edges(self) -> list[tuple[PMFTree, PMFTree]]:
        node_q, res = SimpleQueue(), []
        node_q.put(self.root)
        while not node_q.empty():
            node: PMFTree = node_q.get()
            if hasattr(node, 'lchild'):
                res.append((node, node.lchild))
                node_q.put(node.lchild)
            if hasattr(node, 'rchild'):
                res.append((node, node.rchild))
                node_q.put(node.rchild)
        return res
    
    def get_nodes_labels(self) -> dict[PMFTree, str]:
        nodes, res = self.get_nodes(), {}
        for node in nodes:
            if node.data.name:
                res[node] = node.data.name
        return res

    def get_edges_labels(self):
        return {}
    

@dataclass
class _Plot2DablePMF:
    pmf: PMF
    def get_x(self) -> list:
        assert self.pmf.dimension == 1
        return self.pmf.distribution()
    
    def get_y(self) -> list:
        assert self.pmf.dimension == 1
        return [x[1] for x in self.pmf.distribution_law()]

@dataclass
class _Plot2DableCDF:
    pmf: PMF

    def get_x(self) -> np.ndarray:
        assert self.pmf.dimension == 1
        self.x = np.array(self.pmf.distribution())
        return self.x
    
    def get_y(self) -> np.ndarray:
        assert self.pmf.dimension == 1
        self.y = np.cumsum(np.array([x[1] for x in self.pmf.distribution_law()]))
        return self.y

def graph_pmftree(root: PMFTree):
    gtree: DiGraphable[PMFTree] = _DiGraphablePMFTree(root)
    draw_digraph(gtree)

def plot_pmf(pmf: PMF):
    assert pmf.dimension == 1
    p = _Plot2DablePMF(pmf)
    plot2D(p)

def plot_cdf(pmf: PMF) -> tuple[np.ndarray, np.ndarray]:
    assert pmf.dimension == 1
    plt_cdf = _Plot2DableCDF(pmf, x=np.array([]), y=np.array([]))
    plot2D(plt_cdf)
    return plt_cdf.x, plt_cdf.y

def create_cdf(pmf: PMF) -> tuple[np.ndarray, np.ndarray]:
    assert pmf.dimension == 1
    l = [np.array(pmf.distribution()), np.cumsum(
        np.array([x[1] for x in pmf.distribution_law()]))]
    return tuple([vec.reshape(1, vec.shape[0])[0] for vec in l])