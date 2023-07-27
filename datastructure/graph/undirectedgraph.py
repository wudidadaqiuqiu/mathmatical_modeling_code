from typing import Hashable, TypeVar, Generic, Optional

Num = float | int
Node = TypeVar('Node', bound=Hashable)
Edge = tuple[Node, Node, Num]

class UnDiGraph(Generic[Node], object):
    def __init__(self, nodes: set[Node], edges: list[Edge] | None | dict[Node, list[Num]]=None) -> None:
        super().__init__()
        self.nodes: set[Node] = nodes  # 不变量，node不重复
        _edges: list[Edge] = []
        if isinstance(edges, list | None):  # 不变量，edge唯一
            _edges = edges if edges else []
            self._node_dict: dict[Node, list[tuple[Node, Num]]] = {}
            self.add_edges(_edges)
            return
        assert isinstance(edges, dict)
        for node in self.nodes:
            pass
        raise
        
    def add_nodes(self, nodes: set[Node] | list[Node]):
        self.nodes.update(nodes)
    
    def add_edges(self, edges: list[Edge]):
        for edge in edges:
            self.add_edge(edge)
    
    def add_edge(self, edge: Edge):  # 分解
        node1, node2, num = edge  # 无向图两个端点
        if node1 in self._node_dict:
                self._node_dict[node1].append((node2, num))
        else:
            self._node_dict[node1] = [(node2, num)]
        if node2 in self._node_dict:
                self._node_dict[node2].append((node1, num))
        else:
            self._node_dict[node2] = [(node1, num)]

    def edges(self) -> list[Edge]:
        return [(node, *node_and_num) for node, to_ in self._node_dict.items() for node_and_num in to_]
