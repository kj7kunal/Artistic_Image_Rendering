#!/usr/bin/env python

import VGGNet
import tensorflow as tf
import numpy as np


def init_imG(imC,initial_noise = 0.6):
	_,h,w,c = imC.shape
	imNoise = np.random.uniform(-20,20,(1,h,w,c)).astype('float32')
	return imNoise*initial_noise + imC*(1-initial_noise)


def layer_content_cost(aC,aG):
	m,nh,nw,nc = aG.get_shape().as_list()
	aC_flat = tf.reshape(tf.transpose(aC,perm=[0,3,1,2]),[nc,nh*nw])
	aG_flat = tf.reshape(tf.transpose(aG,perm=[0,3,1,2]),[nc,nh*nw])
	J_content = tf.reduce_sum(tf.square(tf.subtract(aC_flat,aG_flat)))/2
	return J_content

def content_cost(sess,net,imgC):
	sess.run(net['input'].assign(imgC))

	# Get content of content image from the current layer by running session on the chosen layer
	aC = sess.run(net['conv4_2'])
	# Set aG representing content of generated image. Here, a_G references net[lname] and isn't evaluated yet. 
	#Later, image G is assigned as the model input, when we run the session, this will be the activation, with **G** as input.
	aG = net['conv4_2']
	J_content = layer_content_cost(aC,aG)
	return J_content


def Gram_Matrix(A):
	# nc x nc Correlation/Unnormalized cross covariance matrix
	GA = tf.matmul(A,A,transpose_b = True)
	return GA

def layer_style_cost(aS,aG):
	m,nh,nw,nc = aG.get_shape().as_list()
	aS_flat = tf.reshape(tf.transpose(aS,perm=[0,3,1,2]),[nc,nh*nw])
	aG_flat = tf.reshape(tf.transpose(aG,perm=[0,3,1,2]),[nc,nh*nw])
	GS = Gram_Matrix(aS_flat)
	GG = Gram_Matrix(aG_flat)
	J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)/(2*nh*nw*nc)))
	return J_style_layer

def style_cost(sess,net,imgS,style_weights):
	J_style = 0
	sess.run(net['input'].assign(imgS))

	for lname,k in style_weights:
		# Current layer for style image
		intermediate = net[lname]
		# Get style of style image from the current layer by running session on intermediate
		aS = sess.run(intermediate)
		# Set aG representing style of generated image -> The intermediate itself is taken as generated.
		# Here, a_G references net[lname] and isn't evaluated yet. Later, we'll assign the image G as the model input,
        # when we run the session, this will be the activations drawn from the appropriate layer, with **G** as input.
		aG = intermediate
		# Compute cost of current layer
		J_style_layer = layer_style_cost(aS,aG)
		J_style += k*J_style_layer
	return J_style

def total_cost(sess,net,imgC,imgS,style_weights,alpha=10,beta=40):
	J_content = content_cost(sess,net,imgC)
	J_style = style_cost(sess,net,imgS,style_weights)
	return J_content*alpha + J_style*beta
















