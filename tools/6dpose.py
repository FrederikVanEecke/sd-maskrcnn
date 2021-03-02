"""
	Extract 6d pose from depth images using masks obtained by sd-maskrcnn. 

	Author: Frederik Van Eecke 
"""

import os
import sys
sys.path.append("/home/frederik/Documents/GitHub/sd-maskrcnn")

from pose_estimation import detection 
from autolab_core import YamlConfig
from pose_estimation.datasetControl import DatasetHandler
from pose_estimation.pointclouds import MaskedPointclouds, TemplatePointclouds
from pose_estimation.ICP import ICP
import imageio




def obtain_masks(detector, image): 
	results = detector.detect(image.get_image())
	image.set_predictions(results)
	

def load_image(path, file): 
	im = imageio.imread(path+file)
	return im

def main(): 
	pose_config = YamlConfig("/home/frederik/Documents/GitHub/sd-maskrcnn/cfg/6dpose.yaml")
	dataset_config= YamlConfig("/home/frederik/Documents/GitHub/sd-maskrcnn/monkeyData/dataset_generation_params.yaml")

	dataset_path = "/home/frederik/Documents/GitHub/sd-maskrcnn/monkeyData/" #"/home/frederik/Documents/GitHub/sd-maskrcnn/test_dataset/"

	depth_im_path = "images/depth_ims/" 

	ds = DatasetHandler(dataset_path, depth_im_path)
	detector = detection.Detector(pose_config)

	templates = TemplatePointclouds(pose_config, dataset_config)

	temp=templates.get_templates(1)

	masked_pcls = MaskedPointclouds()

	ds_image = ds.load_image()

	_=detector.detect(ds_image)

	ds_image.visualizePreditions()
	
	masked_pcls.feed_image(ds_image)

	icp = ICP(pose_config, dataset_config)

	masked_pcls.render_masked_pointclouds()

	(classes, pointclouds_cleaned) = masked_pcls.get_pointcloudsFor_ICP()

	masked_pcls.render_masked_pointclouds(option='cleaned')

	icp.feed_pointcloudsAndClasses(pointclouds_cleaned, classes)

	masked_pcls.render_templates(temp)

	# _,_,transformations,matching_template = icp.get_best_transformations()
	# icp.render_ipc_results()

	#masked_pcls.render_ICP(matching_template)

	# # print(t)
	# ds_image = ds.load_image() # random image

	# detector = detection.Detector(pose_config)

	# _=detector.detect(ds_image)

	# masked_pcls = MaskedPointclouds()
	# masked_pcls.feed_image(ds_image)
	
	# (classes, pointclouds_cleaned) = masked_pcls.get_pointcloudsFor_ICP()





	#print(matching_template)
	

	# ds_image.visualizePreditions()
	# # results is a dict with keys: 'rois', 'class_ids', 'scores', 'masks', 'time'
	# masked_pcls = MaskedPointclouds()

	# masked_pcls.feed_image(ds_image)
	# # masked_pcls.downsample_pointclouds(voxel_size=0.05)
	# # masked_pcls.clean_pointclouds(downsampled=True)
	# detected_pcds = masked_pcls.get_pointcloudsFor_ICP()
	
	# masked_pcls.render_masked_pointclouds()
	# masked_pcls.downsample_pointclouds(voxel_size=0.05)
	# masked_pcls.clean_pointclouds(downsampled=True)
	# masked_pcls.render_masked_pointclouds(option='cleaned')
	#masked_pcls.render_total_pointcloud()
	#masked_pcls.render_all()

	#get masked poinclouds 

	#print(masks)

	print('Done')
if __name__ == '__main__':
	main()