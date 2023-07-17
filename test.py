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
from matplotlib.animation import FuncAnimation


fig = plt.figure(figsize=(10, 5))  # 创建图
plt.rcParams["font.family"] = "FangSong"  # 支持中文显示
plt.ylim(-12, 12)  # Y轴取值范围
plt.yticks([-12 + 2 * i for i in range(13)], [-12 + 2 * i for i in range(13)])  # Y轴刻度
plt.xlim(0, 2 * np.pi)  # X轴取值范围
plt.xticks([0.5 * i for i in range(14)], [0.5 * i for i in range(14)])  # X轴刻度
plt.title("函数 y = 10 * sin(x) 在[0,2Π]区间的曲线")   # 标题
plt.xlabel("X轴")  # X轴标签
plt.ylabel("Y轴")  # Y轴标签
x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空


def update(n):  # 更新函数
    x.append(n)  # 添加X轴坐标
    y.append(10 * np.sin(n))  # 添加Y轴坐标
    plt.plot(x, y, "r--")  # 绘制折线图


ani = FuncAnimation(fig, update, frames=np.arange(0, 2 * np.pi, 0.1), interval=50, blit=False, repeat=False)  # 创建动画效果
plt.show()  # 显示图片