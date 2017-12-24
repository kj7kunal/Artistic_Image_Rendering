#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import scipy.io


class VGGNet:
	'''
	Class for VGG Network Architecture
	Constructor takes a path to the model mat file.
	Function forward pass takes an image
	'''
	def __init__(self,model_path):
		self.layers = (
			'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
			'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
			'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
			'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
			'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
			'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
			'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
			'relu5_3', 'conv5_4', 'relu5_4')

		VGG = scipy.io.loadmat(model_path)
		self.mean_sub = VGG['meta']['normalization'][0][0][0][0][2][0][0].reshape((1,1,1,3))
		self.weights = VGG['layers'][0]

	def preprocess(self,image):
		#required before sending the image to network
		image = np.reshape(image,((1,)+image.shape))
		return image - self.mean_sub

	def postprocess(self,image):
		#required while saving image
		image = image + self.mean_sub
		return np.clip(image[0],0,255).astype('uint8')


	def forward_pass(self,img=None):
		graph = {}
		_, nh, nw, nc = img.shape

		graph['input'] = tf.Variable(np.zeros((1,nh,nw,nc)),dtype=np.float32)
		X = graph['input']
		for i,l in enumerate(self.layers):
			ltype = l[:4]
			if ltype == "conv":
				W,b = self.weights[i][0][0][2][0]
				W = np.transpose(W, [1, 0, 2, 3])
				b = b.reshape(-1)
				X = tf.nn.bias_add( tf.nn.conv2d(X,tf.constant(W),strides=[1, 1, 1, 1],padding='SAME'), tf.constant(b))
			elif ltype == "relu":
				X = tf.nn.relu(X)
			elif ltype == "pool":
				X = tf.nn.max_pool(X,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
			graph[l] = X

		return graph

# vagg = VGGNet("../../imagenet-vgg-verydeep-19.mat")
# intermediates = vagg.forward_pass()
# print intermediates['input']


