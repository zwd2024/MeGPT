import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
epochs = np.arange(1, 101)
v5 = np.random.normal(0.8, 0.02, 100).cumsum() / np.arange(1, 101)
v5g = np.random.normal(0.78, 0.02, 100).cumsum() / np.arange(1, 101)
v5gd = np.random.normal(0.76, 0.02, 100).cumsum() / np.arange(1, 101)
v5gsd = np.random.normal(0.82, 0.02, 100).cumsum() / np.arange(1, 101)
v5s = np.random.normal(0.75, 0.02, 100).cumsum() / np.arange(1, 101)

# 绘制图形
plt.figure(figsize=(10, 5))
plt.plot(epochs, v5, label='v5.csv', color='blue')
plt.plot(epochs, v5g, label='v5g.csv', color='green')
plt.plot(epochs, v5gd, label='v5gd.csv', color='cyan')
plt.plot(epochs, v5gsd, label='v5gsd.csv', color='orange')
plt.plot(epochs, v5s, label='v5s.csv', color='red')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Precision over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Precision')

# 显示图形
plt.show()