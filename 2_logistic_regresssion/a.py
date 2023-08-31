from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 准备不同的test_size值
# test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
test_sizes = [0.1]

# 用于存储每次迭代的准确率
accuracies = []

# 对每一个test_size值
for test_size in test_sizes:
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 训练模型
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # 预测测试集并计算准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 绘制折线图
plt.plot(test_sizes, accuracies, marker='o')
plt.title('Test Size vs Accuracy')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# 可以用下面的代码，看前100个样本
fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Digit: {digits.target[i]}')

plt.tight_layout()
plt.show()
