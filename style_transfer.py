import vgg
import numpy as np
import tensorflow as tf

from PIL import Image

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def style_transfer(CONTENT_IMG_PATH, STYLE_IMG_PATH, OUTPUT_IMG_PATH, VGG_PATH, iterations, content_weight, style_weight,
		learning_rate, beta1, beta2, epsilon, GPU=-1):
	'''
	Where the magic happens!

	CONTENT_IMG_PATH : Path for the content image
	STYLE_IMG_PATH   : Path for the style image
	OUTPUT_IMG_PATH  : Destination for the output image
	VGG_PATH         : Path for the VGG model
	iterations       : Number of iterations the graph needs to be run for
	content_weight   : Weight of the content image (referred to as alpha in the paper)
	style_weight     : Weight of the style image (referred to as beta in the paper)
	learning_rate    : Learning rate for Adam optimizer
	beta1            : Hyperparameter for Adam optimizer
	beta2            : Hyperparameter for Adam optimizer
	epsilon          : Hyperparameter for Adam optimizer
	GPU              : Integer: -1 for CPU (default); any other non-negative integer for corresponding GPU device
	'''

	if GPU == -1:
		device_prop = "/cpu:0"
	else:
		device_prop = "/gpu:" + str(GPU)

	# Load VGG network
	VGG_net, mean_pixels = vgg.load(VGG_PATH)

	# Read images and preprocess them
	content_img = read_img(CONTENT_IMG_PATH, mean_pixels)
	content_shape = np.shape(content_img)
	style_img = read_img(STYLE_IMG_PATH, mean_pixels)
	style_shape = np.shape(style_img)

	# Compute content features
	content = {}
	with tf.Graph().as_default(), tf.device(device_prop):
		img = tf.constant(content_img, dtype=tf.float32, shape=content_shape)
		net = vgg.build_network(img, VGG_net)

		GPU_config = config=tf.ConfigProto(allow_soft_placement=True) # Suppress error when manually placing device
		with tf.Session(config=GPU_config) as sess:
			for layer in CONTENT_LAYERS:
				content[layer] = net[layer].eval()

	# Compute style features
	style = {}
	with tf.Graph().as_default(), tf.device(device_prop):
		img = tf.constant(style_img, dtype=tf.float32, shape=style_shape)
		net = vgg.build_network(img, VGG_net)

		GPU_config = config=tf.ConfigProto(allow_soft_placement=True) # Suppress error when manually placing device
		with tf.Session(config=GPU_config) as sess:
			for layer in STYLE_LAYERS:
				features = net[layer].eval()
				features = np.reshape(features, (-1, features.shape[3]))
				gram = np.matmul(features.T, features) / features.size
				style[layer] = gram

	# Stylize image
	stylize = tf.Graph()
	with stylize.as_default(), tf.device(device_prop):

		# Initialize output image
		output_img = tf.Variable(tf.random_normal(content_shape))
		net = vgg.build_network(output_img, VGG_net)

		# Content loss
		content_losses = []
		content_blend = {}
		content_blend['relu4_2'] = 1.0
		content_blend['relu5_2'] = 0.0
		for content_layer in CONTENT_LAYERS:
			content_losses.append(content_blend[content_layer] *
				(tf.nn.l2_loss(net[content_layer] - content[content_layer])) / np.asarray(content[content_layer]).size)
		L_content = tf.reduce_sum(content_losses)

		# Style loss
		style_losses = []
		for style_layer in STYLE_LAYERS:
			layer = net[style_layer]
			_, height, width, number = map(lambda i: i.value, layer.get_shape())
			size = height * width * number
			features = tf.reshape(layer, (-1, number))
			gram = tf.matmul(tf.transpose(features), features) / size
			style_losses.append(0.5 * tf.nn.l2_loss(gram - style[style_layer]) / np.asarray(style[style_layer]).size)
		L_style = tf.reduce_sum(style_losses)

		# Total loss
		L_total = content_weight * L_content + style_weight * L_style

		# Optimization
		optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(L_total)

		saver = tf.train.Saver()
		def progress():
			print("\t\tTotal Loss: ", L_total.eval())

		# Run graph
		GPU_config = config=tf.ConfigProto(allow_soft_placement=True) # Suppress error when manually placing device
		with tf.Session(config=GPU_config) as sess:
			tf.global_variables_initializer().run()
			print("Initialized")
			for step in range(iterations):
				optimizer.run()
				if (step % 100 == 0):
					print("\tIteration: ", step)
					progress()
			print("Finished optimizing. Saving checkpoint.")
			save_path = saver.save(sess, "./.ckpt/styleTransfer.ckpt")
			print("Finished saving checkpoint.")

	# Output image
	with tf.Session(graph=stylize) as sess:
		print("Restoring checkpoint.")
		saver.restore(sess, "./.ckpt/styleTransfer.ckpt")
		op_img = tf.cast(output_img, dtype=np.float64).eval()
		print("Restored image.")

	op_img = postprocess(op_img, mean_pixels)
	op_img = Image.fromarray(op_img)
	op_img.save(OUTPUT_IMG_PATH)
	print("Output image generated at %s" %(OUTPUT_IMG_PATH))

def read_img(img_path, mean_pixels):
	'''
	Read image from source
	'''
	img = Image.open(img_path)
	img = np.asarray(img, dtype=np.uint8)
	img = preprocess(img, mean_pixels)
	return img

def preprocess(img, mean_pixels):
	'''
	Preprocess image for computation
	'''
	# Change BGR to RGB
	img = img[...,::-1]
	# Reshape (h, w, d) to (1, h, w, d)
	img = img[np.newaxis,:,:,:]
	img = img - np.array(mean_pixels).reshape((1,1,1,3))
	return img

def postprocess(img, mean_pixels):
	'''
	Process optimized image before rendering it
	'''
	img += np.array(mean_pixels).reshape((1,1,1,3))
	# Reshape (1, h, w, d) to (h, w, d)
	img = img[0]
	img = np.clip(img, 0, 255).astype('uint8')
	# RGB to BGR
	img = img[...,::-1]
	return img
