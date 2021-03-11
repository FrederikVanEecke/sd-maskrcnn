"""
	Author: Frederik Van Eecke 

	Script facilitates detection of object masks ussing sd-maskrcnn. 
"""
import sys 
import os 
sys.path.append("/home/frederik/Documents/GitHub/sd-maskrcnn")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.compat.v1.keras.backend import set_session

from sd_maskrcnn.config import MaskConfig
from mrcnn import model as modellib, utils as utilslib, visualize


class Detector():
	"""
		Detection class 
	"""
	def __init__(self, config): 
		self.config = config 

	
		self.inference_config = MaskConfig(config['model']['settings'])
		
		self.model_dir, _ = os.path.split(config['model']['path'])



	def detect(self, image): 
		"""
			use self.model to obtain masks for input dataset image object, returns the results and sets the predictions for this image object.
		"""
		
		config = tf.ConfigProto(device_count = {'GPU': 0})
		with tf.Session(config=config) as sess:
			set_session(sess)

			model = modellib.MaskRCNN(mode=self.config['model']['mode'], config=self.inference_config, model_dir=self.model_dir)
			print(self.config['model']['path'])
			model.load_weights(self.config['model']['path'], by_name=True)

			results = model.detect([image.get_image()], verbose = 0)
			image.set_predictions(results)
			return results



