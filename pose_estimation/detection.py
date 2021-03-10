"""
	Author: Frederik Van Eecke 

	Script facilitates detection of object masks ussing sd-maskrcnn. 
"""
import sys 
import os 
sys.path.append("/home/frederik/Documents/vintecc/sd-maskrcnn")
print(os.getcwd())

from sd_maskrcnn.config import MaskConfig
from mrcnn import model as modellib, utils as utilslib, visualize


class Detector():
	"""
		Detection class 
	"""
	def __init__(self, config): 
		self.config = config 
		
		inference_config = MaskConfig(config['model']['settings'])
		
		model_dir, _ = os.path.split(config['model']['path'])

		model = modellib.MaskRCNN(mode=config['model']['mode'], config=inference_config, model_dir=model_dir)

		model.load_weights(config['model']['path'], by_name=True)

		self.model = model 


	def detect(self, image): 
		"""
			use self.model to obtain masks for input dataset image object, returns the results and sets the predictions for this image object.
		"""
			
		results = self.model.detect([image.get_image()], verbose = 0)
		image.set_predictions(results)

		return results

	def set_new_weights(path_to_weights): 
		model.load_weights(path_to_weights, by_name=True)
		print('new weights set')


