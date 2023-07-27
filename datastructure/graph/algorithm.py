from .undirectedgraph import UnDiGraph
import numpy as np

ShortestPathGraph = UnDiGraph

def floyd_shortest_path(graph: ShortestPathGraph):
    A, B, node_dict, lenth = *floyd_mat(graph), len(graph.nodes)
    for k in range(lenth):
        for i, j in scan_matrix(A):
            if A[i, k] == None or A[k, j] == None:
                continue
            a = A[i, k] + A[k, j]
            if A[i, j] == None or a < A[i, j]:
                A[i, j] = a
                B[i, j] = B[i, k]
    return A, B, node_dict

def floyd_mat(graph: ShortestPathGraph) -> tuple[np.ndarray, np.ndarray, dict]:
    node_dic, i, lenth = {}, 0, len(graph.nodes)
    for node in graph.nodes:
        node_dic[node] = i
        i += 1
    A, B = [np.array([[None] * lenth] * lenth) for _ in [0, 1]]
    for i in range(lenth):
        A[i, i] = 0
        B[i, i] = i
    for edge in graph.edges():
        A[node_dic[edge[0]], node_dic[edge[1]]] = edge[-1]
        B[node_dic[edge[0]], node_dic[edge[1]]] = node_dic[edge[1]]
    return A, B, node_dic

def scan_matrix(mat: np.ndarray):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            yield i, j