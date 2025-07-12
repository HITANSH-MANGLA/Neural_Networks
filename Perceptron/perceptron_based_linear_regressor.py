import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points = 1200

data=[]
m=0.2
c = 0.5
for i in range(num_points):
    x = np.random.normal(0.0, 0.8)
    noise = np.random.normal(0.0, 0.04)
    y = m * x + c + noise
    data.append([x, y])

x_data = [d[0] for d in data]
y_data = [d[1] for d in data]

plt.plot(x_data, y_data, 'ro')
plt.title('Input Data')
plt.show()

W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

num_iterations = 10
for step in range(num_iterations):
    sess.run(train)
    

    print('\nITERATION', step+1)
    print('W:', sess.run(W)[0])
    print('b:', sess.run(b)[0])
    print('Loss:', sess.run(loss))



    plt.plot(x_data, y_data,'ro')

    plt.plot(x_data, sess.run(W)*x_data + sess.run(b))




    plt.xlabel('dimension 0')
    plt.ylabel('dimension 1')
    plt.title('iteration'+str(step+1)+'of'+str(num_iterations))
    plt.show()
