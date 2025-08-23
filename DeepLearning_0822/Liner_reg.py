# 梯度下降法
# 定义数据特征
x_data = [1,2,3]

# 定义标签
y_data = [2,4,6]

# 初始化参数
w = 4

# 定义线性回归模型
def forward(x):
    return x * w

# 定义损失函数
def loss(xs, ys):
    lossValue = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        lossValue += (y_pred - y) ** 2
    return lossValue / len(xs)

# 定义梯度计算
def gradient(xs, ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

for epoch in range(100):
    cost_val = loss(x_data,y_data)

    grad_val = gradient(x_data,y_data)

    w = w - 0.01 * grad_val

    print('训练轮次：',epoch,'w=',w,'loss=',cost_val)

print("100轮后w已经训练好嘞，学习4小时的的得分为：",forward(4))