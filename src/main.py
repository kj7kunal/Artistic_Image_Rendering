#!/usr/bin/env python

# import scipy.misc
import numpy as np
from PIL import Image
import argparse

import tensorflow as tf

import VGGNet
import stylefunctions as sf

iterations = 1000
learning_rate = 1.0
loss_ratio = 1e-3
noise_ratio = 1.0
path_VGG = "../../imagenet-vgg-verydeep-19.mat"
W = 400.
H = 300.

"""
VGG-19 model: (f,f,n_c,n_f)
-> conv1_1 (3, 3, 3, 64) 	-> relu-> conv1_2 (3, 3, 64, 64)  -> relu-> maxpool
-> conv2_1 (3, 3, 64, 128)	-> relu-> conv2_2 (3, 3, 128, 128)-> relu-> maxpool
-> conv3_1 (3, 3, 128, 256)	-> relu-> conv3_2 (3, 3, 256, 256)-> relu-> conv3_3 (3, 3, 256, 256)-> relu-> conv3_4 (3, 3, 256, 256)-> relu-> maxpool
-> conv4_1 (3, 3, 256, 512)	-> relu-> conv4_2 (3, 3, 512, 512)-> relu-> conv4_3 (3, 3, 512, 512)-> relu-> conv4_4 (3, 3, 512, 512)-> relu-> maxpool
-> conv5_1 (3, 3, 512, 512)	-> relu-> conv5_2 (3, 3, 512, 512)-> relu-> conv5_3 (3, 3, 512, 512)-> relu-> conv5_4 (3, 3, 512, 512)-> relu-> maxpool
-> fullyconnected (7, 7, 512, 4096)-> relu-> fullyconnected (1, 1, 4096, 4096)-> relu-> fullyconnected (1, 1, 4096, 1000)-> softmax
"""

c_layers = 'relu4_2'
c_strength = loss_ratio
s_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
s_strength = 1

ap = argparse.ArgumentParser()
ap.add_argument("-c","--content",required = True, help = "Path to content image")
ap.add_argument("-s","--style",required = True, help = "Path to style image")
ap.add_argument("-o","--output",required = True, help = "Path to rendered output image")
ap.add_argument("-i","--iter", dest = 'iterations', default = iterations, help = "Styling iterations")
ap.add_argument("-p","--path", dest = 'path_VGG', default = path_VGG, help = "Path to VGG Model")
ap.add_argument("-ht","--height", dest = 'H', default = H, help = "Image Height")
ap.add_argument("-wt","--width", dest = 'W', default = W, help = "Image Width")
ap.add_argument("-ss","--styleStrength", dest = 'c_strength', default = c_strength, help = "Style Strength")
ap.add_argument("-cs","--contentStrength", dest = 's_strength', default = s_strength, help = "Content Strength")
ap.add_argument("-sw","--sweights", help = "Style Layer Weights {5 values for 5 layers}")
args = vars(ap.parse_args())

style_weights = args["sweights"]

if style_weights == None:
	style_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
else:
	style_weights = style_weights/np.sum(style_weights)
SW = zip(s_layers,style_weights)

imSize = np.array([W,H]).astype(int)
# imgC = scipy.misc.imread(args["content"])
imgC = Image.open(args["content"])
imgC = np.float32(imgC.resize(imSize,Image.LANCZOS))
print imgC.shape
# imgS = scipy.misc.imread(args["style"])
imgS = Image.open(args["style"])
imgS = np.float32(imgS.resize(imSize,Image.LANCZOS))
print imgS.shape


tf.reset_default_graph()
# g = tf.Graph()
# with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
with tf.Session() as sess:
	VGG = VGGNet.VGGNet(path_VGG)
	
	imgC = VGG.preprocess(imgC)
	imgS = VGG.preprocess(imgS)

	imgG = sf.init_imG(imgC,noise_ratio)

	model = VGG.forward_pass(imgG)


	J = sf.total_cost(sess,model,imgC,imgS,SW,alpha=c_strength,beta=s_strength)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	trainOpt = optimizer.minimize(J)

	sess.run(tf.global_variables_initializer())


	sess.run(model['input'].assign(imgG))

	for i in range(iterations):
		sess.run([trainOpt,J])

		if i%20 == 0:
			# Get Output image for the current iteration
			imgG = sess.run(model['input'])
			print i,sess.run(J)
			save_image = VGG.postprocess(imgG)
			with open(args["output"]+'_'+str(i)+".jpg", 'wb') as file:
				Image.fromarray(save_image).save(file, 'jpeg')


	save_image = VGG.postprocess(imgG)
	with open(args["output"]+"_final.jpg", 'wb') as file:
		Image.fromarray(save_image).save(file, 'jpeg')

