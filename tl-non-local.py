import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tqdm import tqdm
import numpy as np

class NonLocalLayer(Layer):
	def __init__(
		self,
		layer,
		out_channels,
		batchsize, # needs be a placeholder tf.placeholder(tf.int32)
		W_init=tf.truncated_normal_initializer(stddev=0.02),
		b_init=None, #tf.constant_initializer(value=0.0),
		W_init_args=None,
		b_init_args=None,
		sub_sample=True,
		is_bn=False,
		name='NonLocal',
	):
		# check layer name (fixed)
		Layer.__init__(self, layer, name=name)

		# the input of this layer is the output of previous layer (fixed)
		self.inputs = layer.outputs

		# print out info (customized)

		# operation (customized)
		self.batchsize, self.height, self.width, self.in_channels = layer.outputs.get_shape().as_list()
		if self.batchsize is None: self.batchsize = batchsize
		print('  [TL] NonLocalLayer {}: in_channels: {} out_channels: {}'.format(name, self.in_channels, out_channels))
		self.out_channels = out_channels
		if W_init_args is None:
			W_init_args = {}
		if b_init_args is None:
			b_init_args = {}

		with tf.variable_scope(name):
			with tf.variable_scope('g'):
				W_g = tf.get_variable(name='W_conv2d', shape=[1, 1, self.in_channels, self.out_channels],
									initializer=W_init, dtype=tf.float32, **W_init_args)
				if b_init is not None:
					b_g = tf.get_variable(name='b_conv2d', shape=(self.out_channels),
									initializer=b_init, dtype=tf.float32, **b_init_args)
					g = tf.nn.conv2d(self.inputs, W_g, strides=(1, 1, 1, 1), padding='SAME', name='g_conv2d') + b_g
				else:
					g = tf.nn.conv2d(self.inputs, W_g, strides=(1, 1, 1, 1), padding='SAME', name='g_conv2d')

				if sub_sample:
					g = tf.nn.max_pool(g, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='g_subsample')

			with tf.variable_scope('phi'):
				W_phi = tf.get_variable(name='W_conv2d', shape=[1, 1, self.in_channels, self.out_channels],
									initializer=W_init, dtype=tf.float32, **W_init_args)
				if b_init is not None:
					b_phi = tf.get_variable(name='b_conv2d', shape=(self.out_channels),
									initializer=b_init, dtype=tf.float32, **b_init_args)
					phi = tf.nn.conv2d(self.inputs, W_phi, strides=(1, 1, 1, 1), padding='SAME', name='phi_conv2d') + b_phi
				else:
					phi = tf.nn.conv2d(self.inputs, W_phi, strides=(1, 1, 1, 1), padding='SAME', name='phi_conv2d')

				if sub_sample:
					phi = tf.nn.max_pool(phi, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='phi_subsample')

			with tf.variable_scope('theta'):
				W_theta = tf.get_variable(name='W_conv2d', shape=[1, 1, self.in_channels, self.out_channels],
									initializer=W_init, dtype=tf.float32, **W_init_args)
				if b_init is not None:
					b_theta = tf.get_variable(name='b_conv2d', shape=(self.out_channels),
									initializer=b_init, dtype=tf.float32, **b_init_args)
					theta = tf.nn.conv2d(self.inputs, W_theta, strides=(1, 1, 1, 1), padding='SAME', name='theta_conv2d') + b_theta
				else:
					theta = tf.nn.conv2d(self.inputs, W_theta, strides=(1, 1, 1, 1), padding='SAME', name='theta_conv2d')

			g_x = tf.reshape(g, shape=[self.batchsize, self.out_channels, -1])
			g_x = tf.transpose(g_x, [0, 2, 1])
			# g_x.shape: [self.batchsize, -1, self.out_channels]

			phi_x = tf.reshape(phi, shape=[self.batchsize, self.out_channels, -1])
			# phi_x.shape: [self.batchsize, self.out_channels, -1]

			theta_x = tf.reshape(theta, shape=[self.batchsize, self.out_channels, -1])
			theta_x = tf.transpose(theta_x, [0, 2, 1])
			# theta_x.shape: [self.batchsize, -1, self.out_channels]

			f = tf.matmul(theta_x, phi_x)
			f_softmax = tf.nn.softmax(f, -1)
			# f.shape: [self.batchsize, -1, -1]
			y = tf.matmul(f_softmax, g_x)
			y = tf.reshape(y, shape=[self.batchsize, self.height, self.width, self.out_channels])

			with tf.variable_scope('w'):
				W_w = tf.get_variable(name='W_conv2d', shape=[1, 1, self.out_channels, self.in_channels],
									initializer=W_init, dtype=tf.float32, **W_init_args)
				if b_init is not None:
					b_w = tf.get_variable(name='b_conv2d', shape=[self.in_channels],
									initializer=b_init, dtype=tf.float32, **b_init_args)
					w_y = tf.nn.conv2d(y, W_w, strides=(1, 1, 1, 1), padding='SAME', name='w_conv2d') + b_w
				else:
					w_y = tf.nn.conv2d(y, W_w, strides=(1, 1, 1, 1), padding='SAME', name='w_conv2d')
					
				if is_bn:
					# TODO: implement this
					pass

			self.outputs = self.inputs + w_y

		# update layer (customized)
		self.all_layers = list(layer.all_layers)
		self.all_params = list(layer.all_params)
		self.all_drop = dict(layer.all_drop)
		self.all_layers.extend([self.outputs])
		if b_init is not None:
			self.all_params.extend([W_g, b_g, W_phi, b_phi, W_theta, b_theta, W_w, b_w])
		else:
			self.all_params.extend([W_g, W_phi, W_theta, W_w])

def create_one_hot(values):
	import numpy as np
	return np.eye(10)[values]

def build_model(input_tensor, batchsize_tensor):
	net = InputLayer(input_tensor, name='input')
	net = Conv2dLayer(net, act=tf.identity, shape=[3, 3, 1, 32], padding='SAME', name='conv1')
	net = BatchNormLayer(net, act=tf.nn.relu, is_train=True, name='bn1')
	net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='VALID', name='pool1')

	net = NonLocalLayer(net, 32, batchsize_tensor, name='NonLocal1')
	net = Conv2dLayer(net, act=tf.identity, shape=[3, 3, 32, 64], padding='SAME', name='conv2')
	net = BatchNormLayer(net, act=tf.nn.relu, is_train=True, name='bn2')
	net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='VALID', name='pool2')

	net = NonLocalLayer(net, 64, batchsize_tensor, name='NonLocal2')
	net = Conv2dLayer(net, act=tf.identity, shape=[3, 3, 64, 128], padding='SAME', name='conv3')
	net = BatchNormLayer(net, act=tf.nn.relu, is_train=True, name='bn3')
	net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='VALID', name='pool3')

	net = FlattenLayer(net, name='flatten')
	net = DenseLayer(net, 1024, act=tf.nn.relu, name='fc1')
	net = DropoutLayer(net, keep=0.5, name='drop1')
	net = DenseLayer(net, 10, act=tf.identity, name='fc2')

	return tf.nn.softmax(net.outputs, -1), net.outputs, net

def main():
	X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
	y = tf.placeholder(tf.float32, shape=[None, 10])
	batchsize = tf.placeholder(tf.int32)
	pred_softmax, pred_logits, model = build_model(X, batchsize)

	# Loss and train ops
	loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=y))
	optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999).minimize(loss_)
	acc_counter = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(y, 1)), tf.float32))
	acc = acc_counter / tf.cast(batchsize, tf.float32)

	loss_summary = tf.summary.scalar('xentropy loss', loss_)
	acc_summary = tf.summary.scalar('acc', acc)
	summaries = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('{}/{}'.format('log_dir', 'mnist'))

	saver = tf.train.Saver()
	
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
	# Training Phase
	X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
	
	for epoch in range(10):
		n_loss, n_acc, n_batch = 0, 0, 0
		for X_batch, y_batch in tqdm(tl.iterate.minibatches(X_train, y_train, batch_size=4)):
			y_batch = create_one_hot(np.asarray(y_batch))
			fd = {X: X_batch, y: y_batch, batchsize: 4}
			fd.update(model.all_drop)
			_, loss, summary, train_counter = sess.run([optim, loss_, summaries, acc], feed_dict=fd)
			n_batch += 1; n_acc += train_counter; n_loss += loss
		print('Epoch {} of {}: loss: {} acc: {}'.format(epoch, 200, n_loss / n_batch, n_acc / n_batch))
		saver.save(sess, 'pretrained/NonLocal', global_step=epoch)

if __name__ == '__main__':
	# test NonLocalLayer
	# input_x = tf.Variable(tf.random_normal([10, 64, 64, 256]))
	# input_x = InputLayer(input_x)
	# out = NonLocalLayer(input_x, 128)
	# print(out.outputs.get_shape().as_list()) # Expected to (10, 64, 64, 256)
	
	main()