import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datastructure.testdata import TestData, BasicData
from math import sin
def add(a,b):
    return a+b

# a = [1,2,3,4]
# # np.fromfunction(add, ())
# a = pd.DataFrame(a)
# a.plot()

x = np.linspace(0, 2 * np.pi, 200)
# y = np.sin(x).tolist()
y = np.vectorize(sin)(x).tolist()
y2 = np.cos(x).tolist()
# print(x.tolist())
# print(y)
# a = pd.DataFrame([[1,2],[3,4]], index=[2,200])
# a = pd.DataFrame([y,y2], index=x.tolist())
# a = pd.DataFrame(y)
# print(a.index.to_list())
# print(a)

# fig, ax = plt.subplots()
# ax.plot(x,y)
# ax.plot(x,y2)
# # ax.plot(a)
# plt.show()


# b = [1,2]
# c = np.array(b)
# d = np.array(c)
# print(d)
# plt.ion()
# fig, ax = plt.subplots()  # Create a figure containing a single axes.
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
# plt.pause(1)
# plt.ioff()
# i = 0
# print(1)
# while i<10000:
#     print(i)
#     i+=1
# fig.show()

# sorted()
# a, b = TestData.create_data(sin, 0, 6, 50)
# a.show()
# b.show()
# BasicData.plot(a, b)
# np.array()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# fig = plt.figure(figsize=(10, 5))  # 创建图
# plt.rcParams["font.family"] = "FangSong"  # 支持中文显示
# plt.ylim(-12, 12)  # Y轴取值范围
# plt.yticks([-12 + 2 * i for i in range(13)], [-12 + 2 * i for i in range(13)])  # Y轴刻度
# plt.xlim(0, 2 * np.pi)  # X轴取值范围
# plt.xticks([0.5 * i for i in range(14)], [0.5 * i for i in range(14)])  # X轴刻度
# plt.title("函数 y = 10 * sin(x) 在[0,2Π]区间的曲线")   # 标题
# plt.xlabel("X轴")  # X轴标签
# plt.ylabel("Y轴")  # Y轴标签
# x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空


# def update(n):  # 更新函数
#     x.append(n)  # 添加X轴坐标
#     y.append(10 * np.sin(n))  # 添加Y轴坐标
#     plt.plot(x, y, "r--")  # 绘制折线图


# ani = FuncAnimation(fig, update, frames=np.arange(0, 2 * np.pi, 0.1), interval=50, blit=False, repeat=False)  # 创建动画效果
# plt.show()  # 显示图片


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# def random_walk(num_steps, max_step=0.05):
#     """Return a 3D random walk as (num_steps, 3) array."""
#     start_pos = np.random.random(3)
#     steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
#     walk = start_pos + np.cumsum(steps, axis=0)
#     return walk


# def update_lines(num, walks, lines):
#     for line, walk in zip(lines, walks):
#         # NOTE: there is no .set_data() for 3 dim data...
#         # print(walk[:num, :2])
#         print(walk[:num, :2].T)
#         print(walk[:num, 2].shape)
#         line.set_data(walk[:num, :2].T)
#         line.set_3d_properties(walk[:num, 2])
#     return lines


# # Data: 40 random walks as (num_steps, 3) arrays
# num_steps = 3
# walks = [random_walk(num_steps) for index in range(1)]
# print(type(walks[0]))
# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# # Create lines initially without data
# lines = [ax.plot([], [], [])[0] for _ in walks]
# # print(lines[0])

# # Setting the axes properties
# # ax.set(xlim3d=(0, 1), xlabel='X')
# # ax.set(ylim3d=(0, 1), ylabel='Y')
# # ax.set(zlim3d=(0, 1), zlabel='Z')

# # Creating the Animation object
# ani = animation.FuncAnimation(
#     fig, update_lines, num_steps, fargs=(walks, lines), interval=1, blit=True, repeat=False)

# plt.show()
# ani.save("animation.gif", fps=25, writer="imagemagick")

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from functools import partial
# import matplotlib.style as mplstyle
# mplstyle.use('fast')

# fig, ax = plt.subplots()
# line1, = ax.plot([], [], 'ro')

# def init():
#     ax.set_xlim(0, 2000*np.pi)
#     ax.set_ylim(-1, 1)
#     return line1,

# def update(frame, ln, x, y):
#     x.append(frame)
#     y.append(np.sin(frame*0.1))
#     # x = [frame]
#     # y = [np.sin(frame*0.1)]
#     ln.set_data(x, y)
#     # plt.clf()
#     return ln,

# ani = FuncAnimation(
#     fig, partial(update, ln=line1, x=[], y=[]),
#     frames=5000,
#     interval=1,
#     cache_frame_data=False,
#     init_func=init, blit=True)

# plt.show()

# from matplotlib import animation
# print(animation.writers.list())

from datastructure.probability.pmf import PMF, merge_two_pmf, create_pmftree, graph_pmftree
from rich import print
# pm = ProbabilityMass(Point((1,2)), 1)
# print(pm)
# p = Point((1,))
# print(type(p.data))
pmf1 = [((1,),0.2),((2,),0.8)]
pmf2 = [((1,),0.2),((2,),0.8)]
pmf3 = [((1,),0.3),((2,),0.7)]
pmf4 = [((1,),0.3),((2,),0.7)]

# pmf = dict(pmf)
# print(pmf.keys())
pp1, pp2, pp3, pp4 = PMF(pmf1, '1'), PMF(pmf2,'2'), PMF(pmf3,'3'), PMF(pmf4,'4')
# pp1, pp2, pp3, pp4 = PMF(pmf1), PMF(pmf2), PMF(pmf3), PMF(pmf4)
# print(pmf4)
# pp =merge_two_pmf(pp1, pp2)
# print(pp)
p_tree = create_pmftree([pp1, pp2, pp3, pp4])

# print(p_tree)
# print(p_tree.data.is_sum_one())

# graph_pmftree(p_tree)
# dic = {1:2,2:2}
# print([item for item in dic.items()])

# class test(object):
#     def __init__(self) -> None:
#         super().__init__()
#         self.data: int = 0

# print(np.array([test() for i in range(2)]))
# print(test().__dict__)

# a = np.arange(80).reshape((4, 5, 4))
# print(a)

# from dataclasses import dataclass

# @dataclass
# class Test2:
#     a: int
#     b: int | None = None

# print(Test2(1).__dict__)

# import c

# from datastructure.graph.undirectedgraph import UnDiGraph
# from datastructure.graph.algorithm import floyd_shortest_path
# # import sys
# # print(sys.path)
# a: UnDiGraph[int] =UnDiGraph(set([1,2,3,4]), [(1,3, 10), (1,4,60),(2,3,5),(2,4,20),(3,4,1)])
# # print(a.edges())
# a1,b, c = floyd_shortest_path(a)
# print(a1)


# import networkx as nx
# import matplotlib.pyplot as plt
# from pyvis.network import Network

# # Step 1: 构建图的数据结构
# G = nx.Graph()
# edges = [(1, 2, 3), (1, 3, 2), (2, 3, 4), (2, 4, 1), (3, 4, 5)]
# G.add_weighted_edges_from(edges)

# # Step 2: 应用最小生成树算法
# mst_edges = nx.minimum_spanning_edges(G, algorithm='prim', data=False)
# mst = nx.Graph(list(mst_edges))

# # Step 3: 可视化
# # 创建动态网络可视化对象
# nt = Network(notebook=True)

# # 添加原始图的节点和边
# for node in G.nodes():
#     nt.add_node(node)

# for edge in G.edges():
#     nt.add_edge(edge[0], edge[1])

# # 展示原始图
# nt.show("original_graph.html")

# # 添加最小生成树的节点和边，并用动态效果展示生成过程
# nt_bfs = Network(notebook=True)

# for node in G.nodes():
#     nt_bfs.add_node(node)

# for edge in mst.edges():
#     nt_bfs.add_edge(edge[0], edge[1])

# # 以动态效果展示生成最小生成树的过程
# for edge in mst.edges():
#     nt_bfs.highlight(Edge=edge)
#     nt_bfs.show_buttons(filter_=['nodes', 'edges', 'physics'])
#     nt_bfs.show("minimum_spanning_tree.html")


# def c(l):
#     l[0] = 100
#     l.append(20)
#     l.pop(0)
#     return l
# ll = [1,2,3]
# print(c(ll), ll)

from datastructure.graph.directedgraph import DirectedEdge, DirectedGrapph, SPDiGraph
from datastructure.graph.algorithm import bellman_ford_shortest_path
g = DirectedGrapph()
g.add_nodes(range(1, 10))
# print(g.nodes)
g.add_edges([DirectedEdge(*t) for t in [(1,2,6), (1,3,3),(1,4,1),(2,5,1),(3,2,2),(3,4,2),
                                        (4,6,10),(5,4,6),(5,6,4),(5,7,3),(5,8,6),(6,5,10),
                                        (6,7,2),(7,8,4),(9,5,2),(9,8,3)]])
# print(g.edges)
# print(g.to_node_dict)
spg = SPDiGraph(g, 1)
print(*bellman_ford_shortest_path(spg, spg.start))
print(spg.edgeto)
print(spg.dist)
# print(spg.edgeto)
# print(spg.dist)
