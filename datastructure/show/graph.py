from typing import Protocol
# from collections.abc import Hashable
from typing import Hashable, TypeVar, Generic, Iterable
import networkx as nx
import matplotlib.pyplot as plt


Node = TypeVar('Node', bound=Hashable)
Edge = tuple[Node, Node]
Label = str
Num = int | float

class DiGraphable(Generic[Node], Protocol):
    def get_nodes(self) -> list[Node]:
        ...
    def get_edges(self) -> list[Edge]:
        ...
    def get_nodes_labels(self) -> dict[Node, Label]:
        ...
    def get_edges_labels(self) -> dict[Edge, Label]:
        ...
    
def draw_digraph(graph: DiGraphable):
    G = nx.DiGraph()
    G.add_nodes_from(graph.get_nodes())
    G.add_edges_from(graph.get_edges())
    # pos = nx.spring_layout(G)
    pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)
    # pos = nx.circular_layout(G)
    nx.draw(G, pos)

    node_lables = graph.get_nodes_labels()
    edge_labels = graph.get_edges_labels()
    nx.draw_networkx_labels(G, pos, node_lables)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()

class Graphable(Generic[Node], Protocol):
    nodes: Iterable
    edges: Iterable
    def get_edges(self) -> list[tuple[Node, Node, Num]]:
        ...

def draw_graph_with_labels(graph: Graphable, config='ac'):
    G = get_nxDigraph_from(graph, config)
    if G == None:
        return
    # 使用 kamada_kawai_layout 布局算法计算节点的位置布局
    pos = nx.kamada_kawai_layout(G, weight='weight')
    # 绘制带有权重的图，边的长度根据权重来设置
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.show()
    return G

def get_nxDigraph_from(graph: Graphable, config='ac'):
    if config != 'ac':
        return
    G = nx.DiGraph()
    if config[0] == 'a':
        G.add_nodes_from(graph.nodes)
    if config[1] == 'c':
        for edge in graph.get_edges():
            G.add_edge(edge[0], edge[1], weight=edge[2])
    return G