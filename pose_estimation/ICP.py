"""
	implementation of ICP functionalities 

	Author: Frederik Van Eecke
"""
import open3d as o3d
from .pointclouds import TemplatePointclouds, MaskedPointclouds
import numpy as np
from .cam_utils import *
import copy


class ICP():
	"""
		Class encapsulating all ICP related functionalities. An ICP object is related to a specific ds_image. THe ICP object will therefore
		have its own templatePointclouds object linked to it, creating it yourself is therefore only necessary for debugging. 
	"""
	
	def __init__(self, config_6dpose, config_dataset):
		# classes detected
		self.classes=None 
		
		self.templates=TemplatePointclouds(config_6dpose, config_dataset)
		self.pointclouds = MaskedPointclouds()

	def reset(self): 
		self.classes = None
		self.detected_classes = None 
		self.detected_pointclouds = None
		
	def feed_image(self, ds_image): 

		"""
			feed a dataset_image for ICP
			IMAGES SHOULD HAVE BEEN FED TO A DETECTOR FIRST!
		"""

		self.reset()

		self.ds_image = ds_image

		# construcintg correpsonding templates 
		self.templates.feed_image(ds_image)

		# constructing instance pointclouds for ds_image, feeding the image only extracts the instance specific pointclouds
		# before we can do ICP they have to be downsampled on cleaned(background problem)
		self.pointclouds.feed_image(ds_image)

		# get object classes and instance specific pointclouds for ICP 
		(classes, pointclouds_cleaned) = self.pointclouds.get_pointcloudsFor_ICP()

		self.detected_classes = classes
		self.detected_pointclouds = pointclouds_cleaned

		## Both self.detected_classes and self.detected_pointclouds are lists, the i_th element from detected_classes gives the class of the i_th pointclouds in self.detected_pointclouds



	def perform_ICP(self, threshold): 
		"""
			Performs ICP, for this we loop over the detected_pointclouds and use the detected_classes to call the right templates for ICP. 

			sets  two list with best templates and transformations both based on fitness as on rmse. 
			
			templates_fitnessBased = [ [template_trf, icp_trf ] , [template_trf, icp_trf ] ]

												|										|
											    v 										V
				best fitness based template  with transformations            template info for pointcloud 2. 
				  for FIRST pcd from self.detected_pointclouds


			template_trf -> transformed template, template translated towards correspondings pcd's center.
			icp_trf -> transformation obtained by ICP. 

		"""

		# list to store results 
		templates_fitnessBased = []
		templates_rmseBased = []

		count = 1 
		total = len(self.detected_pointclouds)
		for (pcd, pcd_class) in zip(self.detected_pointclouds, self.detected_classes): 
			print('performing ICP on pointcloud {}/{}'.format(count, total))
			count+=1

			# loop over detected (pcd, class) pairs
			# get templates for this pcd
			templates = self.templates.get_templates(class_id = pcd_class)
			print('fitting {} templates'.format(len(templates)))

			best_fitness = 0 # higher = better
			best_rmse = 10**6 # lower = better 

			best_template_fitnessBased = None 
			best_template_rmseBased = None

			# loop over all templates 
			for template in templates:

				# ICP for current combination of template and threshold. 
				transformation, trf_template, fitness, rmse = do_ICP(pcd, template, threshold)

				# check of fitness is better than current best 
				if fitness > best_fitness: 
					best_fitness = fitness
					best_template_fitnessBased = [copy.deepcopy(trf_template), copy.deepcopy(transformation)]

				# check if rmse is better than current best rmse
				if rmse < best_rmse: 
					best_rmse = rmse
					best_template_rmseBased = [copy.deepcopy(trf_template), copy.deepcopy(transformation)]

			# store best templates 
			templates_fitnessBased.append(best_template_fitnessBased)
			templates_rmseBased.append(best_template_rmseBased)


		# make results accesbile 
		self.templates_fitnessBased = templates_fitnessBased
		self.templates_rmseBased = templates_rmseBased
		
		# self.draw_registration_result(pointcloud, templates[0], transform_matrix, transformation)
		# draw_registration_result(pointcloud, templates[0], transformation)


	def render_results(self, option='fitnessBased'):
		"""
			Renders the templates obtained with perform_ICP. 
			option: fitnessBased or rmseBased. -> shows either self.templates_fitnessBased or self.templates_rmseBased
		"""
		intrinsics = self.ds_image.get_intrinsics()
		intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}
		vis = o3d.visualization.Visualizer()

		vis.create_window(width=intrin['w'], height = intrin['h'])

		# target shown
		pcd_bin = self.pointclouds.get_total_image_pointcloud()
		vis.add_geometry(pcd_bin)

		if option == 'fitnessBased': 
			templates = self.templates_fitnessBased
		elif option == 'rmseBased': 
			templates = self.templates_rmseBased
		else: 
			print('not a valid option!')
			print('proceeding with fitnessBased templates.')
			templates = self.templates_fitnessBased


		# add templates. 

		for pcd_array in templates: 
			[template_trf, icp_trf] = pcd_array

			#template_trf.paint_uniform_color([0,1,0])
			#vis.add_geometry(template_trf)

			template_copy = copy.deepcopy(template_trf)
			template_copy.transform(icp_trf)
			template_copy.paint_uniform_color([0,0,1])
			vis.add_geometry(template_copy)
			

		open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
		open3d_intrinsics.set_intrinsics(width=intrin['w'], height=intrin['h'], fx=intrin['fx'], fy=intrin['fy'], cx=intrin['w']/2-0.5, cy=intrin['h']/2-0.5)

		open3d_camera = o3d.camera.PinholeCameraParameters()
		pose = self.ds_image.get_pose()
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]
		open3d_camera.extrinsic = pose
		open3d_camera.intrinsic = open3d_intrinsics

		view_controls  = vis.get_view_control()
		view_controls.convert_from_pinhole_camera_parameters(open3d_camera)
			
		vis.run()
		vis.destroy_window()


	def show_templates(self, class_id, option='r'): 
		"""
			Shows the templates of a specific class as they are, meaning without any transformation. Usefull for debugging purposes. 
			option: 
					-r: show a random one
					-all: show all of them 
		"""

		# get templates 
		templates=self.templates.get_templates(class_id = class_id)

		intrinsics = self.ds_image.get_intrinsics()
		intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}
		

		if option == 'r':
			vis = o3d.visualization.Visualizer()
			vis.create_window(width=intrin['w'], height = intrin['h']) 
			vis.add_geometry(np.random.choice(templates))
			vis.run()
			vis.destroy_window()

		if option =='all': 
			for template in templates: 
				vis = o3d.visualization.Visualizer()
				vis.create_window(width=intrin['w'], height = intrin['h'])
				vis.add_geometry(template)
				vis.run()
				vis.destroy_window()







	  

### SOME FUNCTIONS 

def do_ICP(target_pointcloud, template_pointcloud, threshold):
	"""
		performs ICP to math the template to the target. 
		templates where render at the origin, therefore a translation of the template towards the target_pointclouds center point is useful. 

		target_pointcloud: o3d pointcloud object
		template_pintcloud: o3d pointcloud object
	"""
	trf_template, transform_matrix = getTransformation(target_pointcloud, template_pointcloud)
	reg_p2p = o3d.pipelines.registration.registration_icp(
													    trf_template, target_pointcloud, threshold, np.identity(4),
													    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
													    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000))



	transformation = reg_p2p.transformation
	fitness = reg_p2p.fitness 
	rmse = reg_p2p.inlier_rmse

	return transformation, trf_template, fitness, rmse








    # source_temp = copy.deepcopy(source)
    # target_temp = copy.deepcopy(target)
    # source.paint_uniform_color([0.5, 0.4, 0.8])
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    # source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])