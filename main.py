import os
import argparse
from style_transfer import style_transfer


VGG_PATH = os.path.join('data', 'imagenet-vgg-verydeep-19.mat')
ITERS = 1001
CONTENT_WEIGHT = 50
STYLE_WEIGHT = 400
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
GPU = False

content_path = os.path.join("images", "content_2.jpg")
style_path = os.path.join("images", "style_1.jpg")
output_path = os.path.join("test", "3_output.jpg")


def parse_arguments():
	global content_path, style_path, output_path, VGG_PATH, ITERS
	global CONTENT_WEIGHT, STYLE_WEIGHT, LEARNING_RATE, BETA1, BETA2, EPSILON, GPU

	parser = argparse.ArgumentParser()
	parser.add_argument("--content_img", type=str, help="Path of content image")
	parser.add_argument("--style_img", type=str, help="Path of style image")
	parser.add_argument("--output_img", type=str, help="Destination for output image")
	parser.add_argument("--iterations", type=int, help="Number of iterations of backprop")
	parser.add_argument("--content_weight", type=float, help="Weight of content image (alpha)")
	parser.add_argument("--style_weight", type=float, help="Weight of style image (beta)")
	parser.add_argument("--vgg", type=str, help="Path for VGG model")
	parser.add_argument("--gpu", type=bool, help="Pass in true (1) to utilize GPU computation")
	parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
	parser.add_argument("--beta1", type=float, help="beta1 hyperparameter for Adam optimizer")
	parser.add_argument("--beta2", type=float, help="beta2 hyperparameter for Adam optimizer")
	parser.add_argument("--epsilon", type=float, help="epsilon hyperparameter for Adam optimizer")
	args = parser.parse_args()

	if args.content_img:
		content_path = args.content_img
	if args.style_img:
		style_path = args.style_img
	if args.output_img:
		output_path = args.output_img
	if args.iterations:
		ITERS = args.iterations
	if args.content_weight:
		CONTENT_WEIGHT = args.content_weight
	if args.style_weight:
		STYLE_WEIGHT = args.style_weight
	if args.vgg:
		VGG_PATH = args.vgg
	if args.gpu:
		GPU = args.gpu
	if args.learning_rate:
		LEARNING_RATE = args.learning_rate
	if args.beta1:
		BETA1 = args.beta1
	if args.beta2:
		BETA2 = args.beta2
	if args.epsilon:
		EPSILON = args.epsilon

def main():
	parse_arguments()
	style_transfer(content_path, style_path, output_path, VGG_PATH, ITERS, CONTENT_WEIGHT,
			STYLE_WEIGHT, LEARNING_RATE, BETA1, BETA2, EPSILON, GPU)

if __name__ == '__main__':
	main()
