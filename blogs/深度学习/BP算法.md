---
title: BP算法
date: 2023/12/04
categories:
  - 深度学习
tags:
  - 反向传播
---

```python
# 定义输入输出
x_1 = 40
x_2 = 80
expect_output = 60


# 初始化
w11_1 = 0.5
w12_1 = 0.5
w13_1 = 0.5
w21_1 = 0.5
w22_1 = 0.5
w23_1 = 0.5
w11_2 = 1
w21_2 = 1
w31_2 = 1


# 前向传播
z_1 = x_1 * w11_1 + x_2 * w21_1
z_2 = x_1 * w12_1 + x_2 * w22_1
z_3 = x_1 * w13_1 + x_2 * w23_1
y_pred = z_1 * w11_2 + z_2 * w21_2 + z_3 * w31_2

print(f"预测值为:{y_pred}")

# 计算损失(L2)
loss = 0.5 * (expect_output - y_pred) ** 2

# 计算输出层关于损失函数的梯度
dloss_pred = -(expect_output - y_pred)

# 计算权重关于损失函数的梯度
dloss_w11_2 = dloss_pred * z_1
dloss_w21_2 = dloss_pred * z_2
dloss_w31_2 = dloss_pred * z_3

dloss_w11_1 = dloss_pred * w11_2 * x_1
dloss_w21_1 = dloss_pred * w11_2 * x_2
dloss_w12_1 = dloss_pred * w21_2 * x_1
dloss_w22_1 = dloss_pred * w21_2 * x_2
dloss_w13_1 = dloss_pred * w31_2 * x_1
dloss_w23_1 = dloss_pred * w31_2 * x_2

# 梯度下降法
learning_rate = 1e-5
w11_2 -= learning_rate * dloss_w11_2
w21_2 -= learning_rate * dloss_w21_2
w31_2 -= learning_rate * dloss_w31_2

w11_1 -= learning_rate * dloss_w11_1
w12_1 -= learning_rate * dloss_w12_1
w13_1 -= learning_rate * dloss_w13_1
w21_1 -= learning_rate * dloss_w21_1
w22_1 -= learning_rate * dloss_w22_1
w23_1 -= learning_rate * dloss_w23_1

# 前向传播
z_1 = x_1 * w11_1 + x_2 * w21_1
z_2 = x_1 * w12_1 + x_2 * w22_1
z_3 = x_1 * w13_1 + x_2 * w23_1
y_pred = z_1 * w11_2 + z_2 * w21_2 + z_3 * w31_2

print(f"Final: {y_pred}")


```

