from .undirectedgraph import Node, Num

inf = None
class DirectedEdge(object):
    def __init__(self, v: Node, w: Node, weight: Num) -> None:
        super().__init__()
        assert isinstance(weight, Num)
        self.weight = weight
        self.from_nd = v
        self.to_nd = w

    def __str__(self) -> str:
        return '(' + ''.join([str(x) + ' ' for x in [self.from_nd, self.to_nd, self.weight]]) + ')'
    
    def __repr__(self) -> str:
        return str(self)
    
class DirectedGraph(object):
    def __init__(self) -> None:
        super().__init__()
        self.to_node_dict: dict[Node, list[DirectedEdge]] = {}
        self.from_node_dict = {}
        self.edges: list[DirectedEdge] = []
        self.nodes = set()
        self.V = 0
        self.E = 0
    
    def get_edges(self):
        return [(edge.from_nd, edge.to_nd, edge.weight) for edge in self.edges]

    def add_node(self, node: Node):
        self.nodes.add(node)
        self.V += 1

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def add_edge(self, edge: DirectedEdge):
        assert edge.to_nd in self.nodes and edge.from_nd in self.nodes
        self.edges.append(edge)
        self.E += 1
        self.to_node_dict[edge.from_nd] = self.to_node_dict.get(edge.from_nd, []) + [edge]
        self.from_node_dict[edge.to_nd] = self.from_node_dict.get(edge.to_nd, []) + [edge]

    def add_edges(self, edges):
        for edge in edges:
            assert isinstance(edge, DirectedEdge)
            self.add_edge(edge)

class SPDiGraph(object):
    def __init__(self, digraph: DirectedGraph, start: Node) -> None:
        self.digraph: DirectedGraph = digraph
        assert start in digraph.nodes
        self.start: Node = start
        self.dist = {start : 0}
        self.edgeto: dict[Node, DirectedEdge | None] = {}
        # self.bellman_ford_shortest_path(self.start)
        # for res in self.dijkstra():
        #     pass

    def dist_to(self, v: Node) -> Num | inf:
        return self.dist.get(v, inf)

    def edge_to(self, v: Node):
        return self.edgeto.get(v, None)
    
    def has_path_to(self, v: Node) -> bool:
        return not self.dist_to(v) == inf
    
    def path_to(self, v: Node) -> list[DirectedEdge]:
        res = []
        while (edge := self.edge_to(v)):
            if edge.from_nd == None:
                return
            res.append(edge)
        return res[::-1]
    
    def node_relax(self, v: Node):
        # print(v,'relax')
        if (dist := self.dist_to(v)) == inf:
            return None
        res: list[DirectedEdge] = []
        for to_edge in self.digraph.to_node_dict.get(v, []):
            if to_edge.weight == inf:
                continue
            if (distto := self.dist_to(to_edge.to_nd)) == inf or distto > dist + to_edge.weight:
                self.dist[to_edge.to_nd] = dist + to_edge.weight
                self.edgeto[to_edge.to_nd] = to_edge
                res.append(to_edge.to_nd)
        return res
    
    def dijkstra(self) -> Node:
        q = [self.start]
        while q:
            nd = q.pop()
            yield nd
            if not (nd_l := self.node_relax(nd)):
                continue
            for node in nd_l:
                if node not in q:
                    q.append(node)
            q.sort(key=lambda nd: self.dist_to(nd), reverse=True)
            
        
    def bellman_ford_shortest_path(self, node: Node):
        assert node in self.digraph.nodes
        q, V = [node], self.digraph.V
        for i in range(V):
            qq = []
            while q:
                node = q.pop(0)
                if (relax_nodes := self.node_relax(node)) != None:
                    for relax_node in relax_nodes:
                        if relax_node not in q and relax_node not in qq:
                            qq.append(relax_node)
            q = qq
            if not q:
                return self
        return None