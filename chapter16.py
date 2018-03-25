# Reinforcement learning.
# Import libraries.
import gym

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)
# To plot pretty figures and animations
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from PIL import Image, ImageDraw

try:
    from pyglet.gl import gl_info
    openai_cart_pole_rendering = True   # no problem, let's use OpenAI gym's rendering function
except Exception:
    openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function

def render_cart_pole(env, obs):
    if openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render(mode="rgb_array")
    else:
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000 # Blue Green Red
        pole_col = 0x669acc # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        return np.array(img)

def plot_cart_pole(env, obs):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    img = render_cart_pole(env, obs)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
# Create environment.
env = gym.make('CartPole-v0')

# Returns the first observation.
obs = env.reset()
print(obs)

# Renders the environment.
env.render()

# Render it as a numpy array.
img = env.render(mode='rgb_array')
print(img.shape)

# Ask the environment which actions are possible.
print(env.action_space) # Discrete(2) which means there are two actions (0, 1).

# Accelerate the cart toward the right.
action = 1 # Accelerate right.
obs, reward, done, info = env.step(action)
print(obs, reward, done, info)

# The step() method executes the action and returns four values.
'''
 obs - This is the new observation, the cart is moving towards the right, and the pole
 is still tilted to the right, but its angular velocity is now negative so it will be tilted
 towards the left in the next step.

 reward - This environment gives a reward of 1.0 at every step, the goal is to keep balancing the
 pole for as long as possible.

 done - This is True when the episode is over, this occurs when the pole tilts too much. After which
 the environment must be reset before it can be used again.

 info - This dictionary provides extra debug information in the environments, it should not be used
 for training as it would be cheating.
'''

# Hardcode a policy that accelerates left when the pole is leaning toward the
# left and accelerates right when leaning right, for 500 episodes.
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000): # 1000 would be the highest score.
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
    
# Look at the result.
import numpy as np
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

# Neural network policies.
'''
Takes the observation as input and outputs the operation to be executed.
More precisely, it will estimate a probability for each action, and then we will
select an action randomly according to the estimated probabilities. Because cart 
pole only has two possible outcomes we only need a single output neuron. The output
is a probability of action 0 (left).

We randomly pick an action based on probability to allow our agent to better explore
the feature space. Ie the balance between exploitation and exploration. 

In this environment past states don't have to be considered as there are no hidden 
variables.
'''

# Build the Neural network with tf.
import tensorflow as tf
# 1. Specify the neural network achitecture.
n_inputs = 4 # == env.observation_space.shape[0].
n_hidden = 4 # it's a simple task, we don't need more hidden neurons.
n_outputs = 1 # Only outputs the probability of accelerating left.
# Return a tensor that returns values without scaling for variance.
initializer = tf.contrib.layers.variance_scaling_initializer()

# 2. Build the Neural Network.
'''
It's a simple multi-layer perceptron, with a single output and thus sigmoid.
This gives a probability from 0 to 1. If there are more than 1 possible actions
a softmax should be used.
'''
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
						 kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs,
						 kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

# 3. Select a random action based on the estimated probabilities.
'''
The multinomial() function allows us to pick a random action. This function independently
samples one (or more) integers, given the log probability of each integer. Ie, if you call
it with the array [np.log(0.5), np.log(0.2), np.log(0.3)] and with num_samples = 5 then it will
output 5 integers each with a probability of 50% of being 0, 20% of being 1, 30% of being 2
and so on. We concatenate because we only want one column in this case.
'''
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

# Initialise global variables.
init = tf.global_variables_initializer()

# Initialise the NN to play 1 game.
n_max_steps = 1000
frames = []

with tf.Session() as sess:
	init.run()
	obs = env.reset()
	for step in range(n_max_steps):
		img = render_cart_pole(env, obs)
		frames.append(img)
		action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
		obs, reward, done, info = env.step(action_val[0][0])
		if done:
			break

# Display the video.
env.close()
video = plot_animation(frames)
plt.show()

# New neural network with target probabilities of y, and training operations.
import tensorflow as tf

reset_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.float32, shape=[None, n_outputs])

# Network creation.
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

# For error calculation.
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Make the NN play in 10 different environments in parallel.
n_environments = 10
n_iterations = 1000

# Create the environments.
envs = [gym.make('CartPole-v0') for _ in range(n_environments)]
observations = [env.reset() for env in envs]

with tf.Session() as sess:
	init.run()
	for iteration in range(n_iterations):
		target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations])
		action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
		for env_index, env in enumerate(envs):
			obs, reward, done, info = env.step(action_val[env_index][0])
			observations[env_index] = obs if not done else env.reset()
	saver.save(sess, './my_policy_net_basic.ckpt')

for env in envs:
	env.close()

# Render the policy net.
def render_policy_net(model_path, action, X, n_max_steps=1000):
	frames = []
	env = gym.make('CartPole-v0')
	obs = env.reset()
	with tf.Session() as sess:
		saver.restore(sess, model_path)
		for step in range(n_max_steps):
			img = render_cart_pole(env, obs)
			frames.append(img)
			action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
			obs, reward, done, info = env.step(action_val[0][0])
			if done:
				break
	env.close()
	return frames


frames = render_policy_net('./my_policy_net_basic.ckpt', action, X)
video = plot_animation(frames)
plt.show()

