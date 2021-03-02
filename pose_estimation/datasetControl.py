"""
	Functionalities for interaction with a dataset generatedn using the generate_mask_dataset.py file. 

	Author: Frederik Van Eecke 

	This implementation assumes that the directory structure remained untouched. 
	Structure assumed is 
	dataset/
		images/ 
			depth_ims/
			semantic_masks/
			../
			test_indices.npy
			train_indices.npy
		image_tensors/ 
		state_tensors/

	implementation assumes identical camera intrinsics for all images within a dataset. Different poses are possible 
"""

import os
import sys
sys.path.append("/home/frederik/Documents/GitHub/sd-maskrcnn/maskrcnn")

from autolab_core import YamlConfig, RigidTransform
from perception import CameraIntrinsics
import matplotlib.pyplot as plt 
import imageio
import numpy as np
from mrcnn import visualize

class DatasetHandler(): 
	"""
		Class designed for interaction with by 'generate_mask_dataset.py' generated dataset during the 6d pose estimation pipeline 
	"""
	def __init__(self, pathToDataset, DepthDir, config_filename="dataset_generation_params.yaml"):
		self.datasetDir = pathToDataset
		self.depthDir = DepthDir # full path from pathToDataset on -> probably something like images/depth_ims/ 

		# read the config file within the dataset
		# open config file
		self.config = YamlConfig(pathToDataset+config_filename)

		self.num_states = self.config["num_states"]
		self.images_per_state = self.config["num_images_per_state"]
		self.total_nbr_images = int(self.num_states*self.images_per_state)

		# tensor configurations
		self.states_per_file = self.config['dataset']['states']['tensors']['datapoints_per_file']
		self.images_per_file = self.config['dataset']['images']['tensors']['datapoints_per_file']
		
		# extract intrinsic camera information 
		intrinsics = self.config["state_space"]["cameras"]["general"]
		self.camera_intrinsics = CameraIntrinsics('view_camera', fx=intrinsics['fx'], fy=intrinsics['fy'],
                                          cx=intrinsics['cx'], cy=intrinsics['cy'], skew=0.0,
                                          height=intrinsics['im_height'], width=intrinsics['im_width'])

	def load_image(self, nbr=-1): 
		"""
			method to retrieve a specific (indicated by nbr) depth image from the depth_ims folder. Also the corresponding camera pose is retrieved. 
			set nbr to -1 for random image from dataset
		"""

		assert nbr<self.total_nbr_images, "index to high, max index is {}".format(self.total_nbr_images)
		
		if nbr == -1: 
			nbr = np.random.randint(self.total_nbr_images)

		# path to depth images folder 
		pathToDepth = self.datasetDir+self.depthDir
		# transform number to image_nbr format 
		image_name = 'image_{:06d}.png'.format(nbr)
		# true path of image 
		path = pathToDepth+image_name
		# retrieving image 
		im = imageio.imread(path)

		# get camera pose associated with this specific image
		path_to_poses = 'image_tensors/tensors/'
		camera_pose_npz_file_number = nbr//self.images_per_file
		filename = 'camera_pose_{:05d}.npz'.format(camera_pose_npz_file_number)
		total_path = self.datasetDir + path_to_poses + filename

		d = np.load(total_path)
		matrix = d['arr_0']

		camera_pose_row_within_file = np.mod(nbr, self.images_per_file)

		pose = matrix[camera_pose_row_within_file]
		# isolate rotation and translation parts 
		translation = pose[:3]
		rotation = pose[3:] # quaternions

		R = RigidTransform.rotation_from_quaternion(rotation) # to rotation matrix 
		pose = RigidTransform(R, translation, 'camera', 'world')



		return DatasetImage(im, self.camera_intrinsics, pose)


class DatasetImage(): 



	def __init__(self, im, intrinsics, pose):
		self.image = im 
		# Camera intrinsics object 
		self.intrinsics = intrinsics
		# Rigidtransform object
		self.pose = pose 


	def get_pose(self):
		return self.pose.matrix # return 4*4 matrix representation 
	def get_intrinsics(self): 
		return self.intrinsics
	def get_image(self): 
		return self.image

	def get_masks(self): 
		return self.masks

	def get_nbr_objects_detected(self): 
		return len(self.predictions['scores'])

	def get_rois(self): 
		return self.predictions['rois']

	def get_classes(self): 
		return self.predictions['class_ids']

	def set_predictions(self, predictions):
		self.predictions = predictions[0]
		# set masks seperatly for quick access	
		self.masks = self.predictions['masks']



	def visualizePreditions(self): 
		fig = plt.figure(figsize=(1.7067, 1.7067), dpi=300, frameon=False)
		ax = plt.Axes(fig, [0.,0.,1.,1.])
		fig.add_axes(ax)
		r = self.predictions
		visualize.display_instances(self.image, r['rois'], r['masks'], r['class_ids'], ['bg', 'obj'], ax=ax, scores=None, show_bbox=False, show_class=False)

		file_name = os.path.join(os.getcwd(), 'visualization')
		fig.savefig(file_name, transparent=True, dpi=300)
		print('saved to: {}'.format(os.getcwd()))
		plt.close()