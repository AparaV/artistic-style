import os
import argparse
from style_transfer import style_transfer

'''
Argument parser
 - For CPU/GPU computation
 - images: content, style and output location
 - iterations
 - content and style weights
 - model path
'''

VGG_PATH = os.path.join('data', 'imagenet-vgg-verydeep-19.mat')
ITERS = 1001
CONTENT_WEIGHT = 50
STYLE_WEIGHT = 400
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08

content_path = os.path.join("test", "2_content.jpg")
style_path = os.path.join("images", "style_1.jpg")
output_path = os.path.join("test", "2_output.jpg")

def main():
	style_transfer(content_path, style_path, output_path, VGG_PATH, ITERS, CONTENT_WEIGHT,
			STYLE_WEIGHT, LEARNING_RATE, BETA1, BETA2, EPSILON, GPU=True)

if __name__ == '__main__':
	main()
