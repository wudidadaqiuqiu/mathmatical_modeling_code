from typing import Protocol
from collections.abc import Hashable
import networkx as nx
import matplotlib.pyplot as plt


Node = Hashable
Edge = tuple[Node, Node]
Label = str

class DiGraphable(Protocol):
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
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    node_lables = graph.get_nodes_labels()
    edge_labels = graph.get_edges_labels()
    nx.draw_networkx_labels(G, pos, node_lables)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()

