import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.testdata import TestData, BasicData
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

from matplotlib import animation
print(animation.writers.list())