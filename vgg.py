import os
import scipy.io
import numpy as np
import tensorflow as tf

def load(path):
	raw = scipy.io.loadmat(path)
	net = raw['layers'][0]
	mean_pixels = raw['meta'][0][0][2][0][0][2][0][0]
	return net, mean_pixels

def build_network(image, vgg):
	'''
	Builds the network by computing convolutions and activations at each stage.
	Done explicitly for readability.

	Network Layer Structure
		conv1_1, relu1_1, conv1_2, relu1_2, pool1
		conv2_1, relu2_1, conv2_2, relu2_2, pool2
		conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3, conv3_4, relu3_4, pool2
		conv4_1, relu4_1, conv4_2, relu4_2, conv4_3, relu4_3, conv4_4, relu4_4, pool2
		conv5_1, relu5_1, conv5_2, relu5_2, conv5_3, relu5_3, conv5_4, relu5_4, pool2
	'''

	net = {}
	net['input'] = image

	# Layer 1
	W, b = get_weights(vgg, 0)
	net['conv1_1'] = convolution(net['input'], W, b)
	net['relu1_1'] = relu(net['conv1_1'])
	W, b = get_weights(vgg, 2)
	net['conv1_2'] = convolution(net['relu1_1'], W, b)
	net['relu1_2'] = relu(net['conv1_2'])
	pooling = pooling_type(vgg, 4)
	net['pool1'] = pool(net['relu1_2'], pooling)

	# Layer 2
	W, b = get_weights(vgg, 5)
	net['conv2_1'] = convolution(net['pool1'], W, b)
	net['relu2_1'] = relu(net['conv2_1'])
	W, b = get_weights(vgg, 7)
	net['conv2_2'] = convolution(net['relu2_1'], W, b)
	net['relu2_2'] = relu(net['conv2_2'])
	pooling = pooling_type(vgg, 9)
	net['pool2'] = pool(net['relu2_2'], pooling)

	# Layer 3
	W, b = get_weights(vgg, 10)
	net['conv3_1'] = convolution(net['pool2'], W, b)
	net['relu3_1'] = relu(net['conv3_1'])
	W, b = get_weights(vgg, 12)
	net['conv3_2'] = convolution(net['relu3_1'], W, b)
	net['relu3_2'] = relu(net['conv3_2'])
	W, b = get_weights(vgg, 14)
	net['conv3_3'] = convolution(net['relu3_2'], W, b)
	net['relu3_3'] = relu(net['conv3_3'])
	W, b = get_weights(vgg, 16)
	net['conv3_4'] = convolution(net['relu3_3'], W, b)
	net['relu3_4'] = relu(net['conv3_4'])
	pooling = pooling_type(vgg, 18)
	net['pool3'] = pool(net['relu3_4'], pooling)

	# Layer 4
	W, b = get_weights(vgg, 19)
	net['conv4_1'] = convolution(net['pool3'], W, b)
	net['relu4_1'] = relu(net['conv4_1'])
	W, b = get_weights(vgg, 21)
	net['conv4_2'] = convolution(net['relu4_1'], W, b)
	net['relu4_2'] = relu(net['conv4_2'])
	W, b = get_weights(vgg, 23)
	net['conv4_3'] = convolution(net['relu4_2'], W, b)
	net['relu4_3'] = relu(net['conv4_3'])
	W, b = get_weights(vgg, 25)
	net['conv4_4'] = convolution(net['relu4_3'], W, b)
	net['relu4_4'] = relu(net['conv4_4'])
	pooling = pooling_type(vgg, 27)
	net['pool4'] = pool(net['relu4_4'], pooling)

	# Layer 5
	W, b = get_weights(vgg, 28)
	net['conv5_1'] = convolution(net['pool4'], W, b)
	net['relu5_1'] = relu(net['conv5_1'])
	W, b = get_weights(vgg, 30)
	net['conv5_2'] = convolution(net['relu5_1'], W, b)
	net['relu5_2'] = relu(net['conv5_2'])
	W, b = get_weights(vgg, 32)
	net['conv5_3'] = convolution(net['relu5_2'], W, b)
	net['relu5_3'] = relu(net['conv5_3'])
	W, b = get_weights(vgg, 34)
	net['conv5_4'] = convolution(net['relu5_3'], W, b)
	net['relu5_4'] = relu(net['conv5_4'])
	pooling = pooling_type(vgg, 36)
	net['pool5'] = pool(net['relu5_4'], pooling)

	return net


'''
Structure of the VGG net:

VGG_net[x][0][0][y][0]

x: Layer number

y = 0: Name of layer
y = 1: Type of layer {conv, relu, pool}
y = 2: Value in layer {weights and biases, 0, type of pooling}
'''

def get_weights(vgg, layer):
	'''
	Returns the weights and biases for each convolution layer.
	'''
	weights = tf.constant(vgg[layer][0][0][2][0][0])
	temp = vgg[layer][0][0][2][0][1]
	biases = tf.constant(np.reshape(temp, (temp.size)))
	return weights, biases

def pooling_type(vgg, layer):
	'''
	Finds the type of pooling at each pooling layer
	'''
	return vgg[layer][0][0][2][0]

def convolution(layer_input, weights, biases):
	'''
	Performs a convolution operation
	'''
	conv = tf.nn.conv2d(layer_input, weights, [1, 1, 1, 1], padding='SAME')
	conv = conv + biases
	return conv

def relu(layer_input):
	'''
	Performs an activation with ReLU
	'''
	reLU = tf.nn.relu(layer_input)
	return reLU

def pool(layer_input, pooling_type):
	'''
	Performs pooling operation
	'''
	if pooling_type == 'max':
		pooling = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	else:
		pooling = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return pooling
