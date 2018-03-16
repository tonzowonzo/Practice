# Chapter 9 - Up and running with Tensorflow.
import tensorflow as tf

# Simple graph and running it in a session.
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

# Run the graph
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

# Run it inside a session.
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    
# Run with global variable initialisation.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run() # Initialise all the vars
    result = f.eval()
    
# Can also run a session like this, but it needs to be closed.
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

'''
A tf program is split into two parts typically: firstly we build a computational
graph (construction phase) and then we run it (execution phase).
'''

# Managing graphs.
# Any node you create is automatically added to the default graph.
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

# If you don't want to do this you need to make a temporary graph.
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
    
x2.graph is graph

x2.graph is tf.get_default_graph()

# Clearing graphs
tf.reset_default_graph()

# Lifecycle of a node value.
'''
When you evaluate a node, tf automatically determines the set of nodes that it
depends on and evaluates those first ie:
'''
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15
    
# Cause y and z to hold their values.
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)
    
# Linear regression with tensorflow.
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X) # X transposed.
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y) # Normal equation.

# Implementing gradient descent.
# Normalise housing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_housing_data_plus_bias = sc.fit_transform(housing_data_plus_bias)

# Manually computing the gradients.
'''
random_uniform() creates a node in the graph that will generate a tensor 
containing random values, given its shape and value range.

assign() creates a node that will assign a new value to a variable. In this
case, it implements the batch gradient descent step theta(next step) = theta - 
n deltatheta MSE(theta).

The main loop executes the training step n_epochs times, every 100 iters it 
prints out the current MSE.
'''


n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
    
