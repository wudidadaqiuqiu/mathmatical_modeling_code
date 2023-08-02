from ..interator import IterQueue, brofir_iter_plus, improved_brofir_iter
from .directedgraph import DirectedEdge, DirectedGraph, SPDiGraph
from .undirectedgraph import UnDiGraph, Node
import numpy as np

ShortestPathGraph = UnDiGraph
ShortestPathNode = Node

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

# def bellman_ford_shortest_path(graph: SPDiGraph, node: Node):
#         assert node in graph.digraph.nodes
#         q, V, i, cycle_in = IterQueue([node]), graph.digraph.V, 0, []
#         def one_round(qe: IterQueue): # 一轮放松
#             nonlocal i, cycle_in
#             if i == V:  # V轮结束, 算法结束
#                 cycle_in = q.pop_queue()
#                 return []
#             i += 1
#             return sum([relaxed_nodes for item in qe.pop_queue(  # 取出上一轮放松成功的节点, 进行松弛操作
#                     ) if (relaxed_nodes := graph.node_relax(item))], [])
#         def added_condition(nd: Node, qe: IterQueue):  # 将不在队列中的节点加入队列
#             return False if nd in qe.get_queue() else True
#         for node in improved_brofir_iter(q, one_round, added_condition, q):  # 广度搜素
#             pass
#         return cycle_in, i  # 队列还有没剩, 第几轮结束

def bellman_ford_shortest_path(graph: SPDiGraph, node: Node):
        assert node in graph.digraph.nodes
        q, V, i, cycle_in = IterQueue([node, None]), graph.digraph.V, 0, []
        def relax(nd: Node | None):
            nonlocal i, V, cycle_in
            if nd == None:
                i += 1
                return [None] if not q.empty() else []
            if i == V:
                cycle_in = q.pop_queue() # 结束
                return []
            return relaxnodes if (relaxnodes := graph.node_relax(nd)) else []
        def added(nd: Node | None, qe: IterQueue):  # 将不在队列中的节点加入队列
            if nd == None:
                return (True, None)
            return (False, None) if nd in qe.get_queue() else (True, nd)
        for node in brofir_iter_plus(q, relax, added, q):  # 广度搜素
            pass
        return cycle_in, i  # 队列还有没剩, 第几轮结束