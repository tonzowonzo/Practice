# Chapter 9 - Up and running with Tensorflow.
import tensorflow as tf
import os

# Set working dir
os.chdir(r'C:/Users/Tim/pythonscripts/MLbook')
# Simple graph and running it in a session.
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

# Reset graph function.
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
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
    
# Using autodiff (automatically calculate differntial eqs)
'''
tf.gradients takes an operation (mse) and a list of variables (theta) and creates
a list of ops to compute the gradients of the op with regards to each var. So the 
gradients node will compute the gradient vector of the MSE with regards to theta.

There are four ways to autodiff:
    1. Numerical differentiation (low accuracy)
    2. Symbolic differentiation (high accuracy)
    3. Forward-mode autodiff (high accuracy)
    4. Reverse-mode autodiff (high accuracy (one used in tensorflow))
'''
gradients = tf.gradients(mse, [theta])[0]

# Using an optimizer to do gradient descent.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Using a different optimizer.
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9)
training_op = optimizer.minimize(mse)
# Initiate the global variables.
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        sess.run(training_op)
        
    best_theta = theta.eval()
    
print('Best theta:')
print(best_theta)

# Feeding data to the training algorithm.
'''
Implementing mini batch gradient descent. X and y need to be replaced at every
iteration with the next mini batch. The easiest way to do this is to use placeholder
nodes. These nodes are special because they don't perform any computation, they only 
output data you tell them to output at runtime.

To create a placeholder you use the placeholder() func and specify the output
tensors dtyle. Optionally you can also specify a shape, None means any size.
'''
# For example:
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)

# Mini batch gradient descent.
n_epochs = 1000
learning_rate = 0.01
reset_graph()

# Define placeholders.
X = tf.placeholder(tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

# Constants.
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

# fetch the batches of x and y data.
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch
    
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
    best_theta = theta.eval()
    
print(best_theta)

# Saving and restoring a model.
'''
To save in tensorflow, you need to create a Saver at the end of construction
phase and at the beginning of the execution phase you don't initialise with an
init node, you call the restore() method of the saver object instead.
'''
reset_graph()
n_epochs = 1000
learning_rate = 0.01


X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")           
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")           
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                 
error = y_pred - y                                                                  
mse = tf.reduce_mean(tf.square(error), name="mse")                                  
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)        
training_op = optimizer.minimize(mse)                                                 

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Save the model
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
            save_path = saver.save(sess, '/tmp/my_model.ckpt')
        sess.run(training_op)
        
    best_theta = theta.eval()
    save_path = saver.save(sess, '/tmp/my_model_final.ckpt')
    
print(best_theta)

# Restore the model
with tf.Session() as sess:
    saver.restore(sess, '/tmp/my_model_final.ckpt')
    best_theta_restored = theta.eval()
    
# Visualising the graph

from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
    
show_graph(tf.get_default_graph())

# Using tensorboard.
reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

# For writing to the tensorboard summary
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:                                                        
    sess.run(init)                                                             

    for epoch in range(n_epochs):                                             
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()   

file_writer.close()
print(best_theta)                                              
