# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
'''进行数据的构造'''
num_point = 1000
vector = []

for i in range(num_point):
    x = np.random.normal(0.0,1.5)
    y = 0.4*x+2.6+np.random.normal(0.0,0.05)
    vector.append([x,y])
x_data = [v[0] for v in vector]
y_data = [v[1] for v in vector]

plt.scatter(x_data,y_data,c = 'b')
#plt.show()


'''进行线性回归'''
w = tf.Variable(tf.random_uniform([1],-1.0,1.0),name = 'W')
b = tf.Variable(tf.zeros([1]),name = 'b')
y = w*x_data+b
loss = tf.reduce_mean(tf.square(y-y_data),name = 'loss')
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss,name='train')
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
# 初始化的W和b是多少
print ("W =", sess.run(w), "b =", sess.run(b), "loss =", sess.run(loss))
# 执行20次训练
for step in range(500):
    sess.run(train)
    # 输出训练好的W和b
    print ("W =", sess.run(w), "b =", sess.run(b), "loss =", sess.run(loss))
writer = tf.train.SummaryWriter("./tmp", sess.graph)

plt.plot(x_data,sess.run(w)*x_data+sess.run(b),c = 'r')
plt.show()
