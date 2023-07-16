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
a, b = TestData.create_data(sin, 0, 6, 50)
a.show()
b.show()
BasicData.plot(a, b)
# np.array()