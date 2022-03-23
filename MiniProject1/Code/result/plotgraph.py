import matplotlib.pyplot as plt
import numpy as np


a = open("bs-64 lr-0.100000 -ds30.txt", "r").readlines()
b = open('bs-64 lr-0.100000 -ds40.txt', 'r').readlines()
c = open('bs-64 lr-0.100000 -ds50.txt', 'r').readlines()
d = open('bs-64 lr-0.100000 -ds60.txt', 'r').readlines()
#将txt文件转换成数组形式保存
for fields1 in a:
    fields1 = fields1.strip()
    fields1 = fields1.strip("[]")
    fields1 = fields1.split(", ")

for fields2 in b:
    fields2 = fields2.strip()
    fields2 = fields2.strip("[]")
    fields2 = fields2.split(", ")

for fields3 in c:
    fields3 = fields3.strip()
    fields3 = fields3.strip("[]")
    fields3 = fields3.split(", ")

for fields4 in d:
    fields4 = fields4.strip()
    fields4 = fields4.strip("[]")
    fields4 = fields4.split(", ")


fields1 = [float(x) for x in fields1]
fields2 = [float(x) for x in fields2]
fields3 = [float(x) for x in fields3]
fields4 = [float(x) for x in fields4]
X = np.linspace(0, 200, 200)
Y1 = np.array(fields1)
Y2 = np.array(fields2)
Y3 = np.array(fields3)
Y4 = np.array(fields4)

plt.xlim(0,210)
plt.ylim(80,100)
plt.plot(X, Y1, label = '1')
plt.plot(X, Y2, label = '2')
plt.plot(X, Y3, label = '3')
plt.plot(X, Y4, label = '4')
plt.legend()
plt.show()
