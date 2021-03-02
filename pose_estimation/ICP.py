"""
	implementation of ICP functionalities 

	Author: Frederik Van Eecke
"""
import open3d as o3d
from .pointclouds import TemplatePointclouds
import numpy as np
from .cam_utils import *
import copy

class ICP():
	
	def __init__(self, config_6dpose, config_dataset):
		self.classes=None 
		self.pointclouds=None 
		self.templates=TemplatePointclouds(config_6dpose, config_dataset)

	def feed_pointcloudsAndClasses(self, pointclouds, classes):
		self.classes=classes
		self.pointclouds=pointclouds


	def get_best_transformations(self): 
		threshold = 0.02

		fitnesses = []
		rmse = []
		transformations = []
		matchingTemplate = []

		print('performing ICP:')
		for c, pcd in zip(self.classes,self.pointclouds):
			templates = self.templates.get_templates(c)
			best_fitness = -1 
			current_transformation = None
			current_rmse = None
			current_template = None

			for template in templates: 
				trf_template = getTransformation(pcd, template)
				reg_p2p = o3d.pipelines.registration.registration_icp(
													    trf_template, pcd, threshold, np.identity(4),
													    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
													    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
				if reg_p2p.fitness > best_fitness:
					best_fitness = reg_p2p.fitness 
					current_transformation = reg_p2p.transformation
					current_rmse = reg_p2p.inlier_rmse
					current_template = template
			
				
			fitnesses.append(best_fitness)
			rmse.append(current_rmse)
			transformations.append(current_transformation)
			matchingTemplate.append(current_template)

		self.matchingTemplates = matchingTemplate
		self.transformations = transformations

		return fitnesses, rmse, transformations, matchingTemplate

	def render_ipc_results(self):
		i = 0 
		for pcd  in self.pointclouds:
			trf_template = getTransformation(pcd, self.matchingTemplates[i])
			source_temp = copy.deepcopy(trf_template)
			target_temp = copy.deepcopy(pcd)
			source_temp.paint_uniform_color([1, 0.706, 0])
			target_temp.paint_uniform_color([0, 0.651, 0.929])
			source_temp.transform(self.transformations[i])
			o3d.visualization.draw_geometries([source_temp, target_temp],zoom=0.4459,front=[0.9288, -0.2951, -0.2242],
												lookat=[1.6784, 2.0612, 1.4451],up=[-0.3402, -0.9189, -0.1996])
			i+=1 

	    











