import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 读取数据
dataset  = pd.read_csv("breast_cancer.csv")

# 提取x
X =  dataset.iloc[:, :, -1]

# 提取数据中的标签
Y = dataset['target']

# 划分数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

# 进行数据的归一化
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# 逻辑回归模型的搭建
lr = LogisticRegression()
lr.fit(x_train, y_train)

# 利用训练好的模型进行推理测试
pre_result = lr.predict(x_test)