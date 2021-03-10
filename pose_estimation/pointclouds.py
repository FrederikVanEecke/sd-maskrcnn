"""
	functionalities facilitating the usage and manipulation of point clouds. both ground thruth point cloud as masked poinclouds

	Author: Frederik Van Eecke
"""

import os
import sys
sys.path.append("/home/frederik/Documents/vintecc/sd-maskrcnn")
from .cam_utils import * 

import open3d as o3d
import PIL
from PIL import Image
import cv2
from perception import DepthImage
from sklearn.cluster import DBSCAN 
from collections import Counter
import matplotlib.pyplot as plt 
import time

import pybullet 
from pyrender import (Scene, IntrinsicsCamera, Mesh, DirectionalLight, Viewer,
                      MetallicRoughnessMaterial, Node, OffscreenRenderer, RenderFlags)

from scipy.spatial.transform import Rotation as R
# from open3d.geometry.PointCloud import create_from_depth_image
# from open3d.camera import PinholeCameraIntrinsic 


class TemplatePointclouds(): 
	"""
		Class for constructing/retrieving object templates correpsonding to found maskedPointcloud objects. 
	"""
	def __init__(self, config_6dpose, config_dataset):
		self.dataset_config = config_dataset
		self.pose_config = config_6dpose
		self.nbr_of_viewpoints = self.pose_config['pointcloud_templates']['viewpoints']
		self.nbr_of_view_axis_rotations = self.pose_config['pointcloud_templates']['view_axis_rot']


		self.meshes_dir = self.dataset_config['urdf_cache_dir'] 

		if not os.path.isabs(self.meshes_dir):
			self.meshes_dir = os.path.join(os.getcwd(), '..',  self.meshes_dir)

		


		# templates related to this ds_image are stored as a dictionnary. Templates are created for each class recognized within the ds_image 
		# templates are added in the following manner: self.templates[class] = [templ1, templ2, ..., templ_n]


	

		self.mapping = {1:'ycb~cubei7'} # temporary mapping for 1 cube case. 
		#self.mapping = {1: 'ycb~monkeyHead'}

		self.ds_image = None
		self.templates = None

	def reset(self):
		self.ds_image = None
		self.templates = None



	def feed_image(self, ds_image):
		"""
			Feeding a dataset image to the template object. 
		"""
		self.reset()
		self.ds_image = ds_image
		self.templates = dict()
		self.create_templates()



	def create_templates(self):
		"""
			Creates templates for every class predicted to be presented within the ds_image. 
		"""

		intrinsics = self.ds_image.get_intrinsics()

		fx=intrinsics.fx
		fy=intrinsics.fy
		ppx=intrinsics.cx
		ppy=intrinsics.cy
		w=intrinsics.width 
		h=intrinsics.height

		far=1.118
		near=0.458
		opengl_mtx = np.array([
							[2*fx/w, 0.0, (w - 2*ppx)/w, 0.0],
							[0.0, 2*fy/h, -(h - 2*ppy)/h, 0.0],
							[0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
							[0.0, 0.0, -1.0, 0.0]
							]).T.reshape(-1).tolist()

		pose = self.ds_image.get_pose()
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]

		unique_classes = np.unique(self.ds_image.get_classes())
		obj_t = np.array([0,0,0.05]) # objects fixed at the origin. 

		for class_id in unique_classes: 
			depth_im = []
			class_templates = [] # array to store class specific pointclouds
			# get class urdf file 
			file = os.path.join(self.meshes_dir, '{}/{}.urdf'.format(self.mapping[class_id], self.mapping[class_id]))

			# connect to pybullet: 
			pybullet.connect(pybullet.DIRECT) # change to DIRECT/GUI when done debugging. 
			q=pybullet.getQuaternionFromEuler(np.random.uniform(low=[0]*3,high=[0]*3))
			print(file)
			obj = pybullet.loadURDF(file, 
									obj_t,
									q, 
									useFixedBase=False)

			# bak = pybullet.loadURDF("/home/frederik/Documents/vintecc/sd-maskrcnn/datasets/oneCube/urdf/cache/bin/bin.urdf", 
			# 						np.zeros(3), 
			# 						pybullet.getQuaternionFromEuler(np.random.uniform(low=[0]*3,high=[0]*3)), 
			# 						useFixedBase=True
			# 						)

			# build images with complitely random orientations
			for i in range(self.nbr_of_viewpoints): 
				# creatre viewpoints of a specific object class 
				data_cam=pybullet.getCameraImage(w,h,pose.T.reshape(-1).tolist(),opengl_mtx)

				depthSample = 2.0 * data_cam[3]- 1.0;
				zLinear = 2.0 * near * far / (far + near - depthSample * (far - near)) 
				im=np.array(zLinear*255,dtype=np.float64)
				
				depth_im.append(im) # get depth image

				# show_im=Image.fromarray(im)
				# show_im.show()
				idx = np.where(im == np.max(im))
				im[idx] = np.nan

				pcl = intrinsics.deproject(DepthImage(im.astype('float'), frame='view_camera')) # retrieve pointcloud using camera intrinsics 

				pcd = o3d.geometry.PointCloud()
				pcd.points = o3d.utility.Vector3dVector(np.array(pcl.data).T)
				ds_pcd = pcd.voxel_down_sample(0.01)

				class_templates.append(ds_pcd)

				# reset to random orientation for new viewpoint
				q=pybullet.getQuaternionFromEuler(np.random.uniform(low=[-np.pi]*3,high=[np.pi]*3))
				obj_t = np.array([np.random.uniform(0,0.05), np.random.uniform(0,0.05), 0.05])
				pybullet.resetBasePositionAndOrientation(obj,obj_t,q)
				pybullet.stepSimulation()

			# build images with rotations along view axis. 
			rotationAxis = np.array([0,0,1])
			rotationUnit = (2*np.pi)/(self.nbr_of_view_axis_rotations)
			RotationAngle = 0

			for i in range(self.nbr_of_view_axis_rotations): 
				RotationAngle = RotationAngle + rotationUnit
				#q=pybullet.getQuaternionFromEuler(np.random.uniform(low=[-np.pi]*3,high=[np.pi]*3))

				x = rotationAxis[0] * np.sin(RotationAngle / 2)
				y = rotationAxis[1] * np.sin(RotationAngle / 2)
				z = rotationAxis[2] * np.sin(RotationAngle / 2)
				ww = np.cos(RotationAngle / 2)
				q = np.array([x,y,z,ww])
				obj_t = np.array([np.random.uniform(0,0.05), np.random.uniform(0,0.05), 0.05])
				pybullet.resetBasePositionAndOrientation(obj,obj_t,q)

				data_cam=pybullet.getCameraImage(w,h,pose.T.reshape(-1).tolist(),opengl_mtx)

				depthSample = 2.0 * data_cam[3]- 1.0;
				zLinear = 2.0 * near * far / (far + near - depthSample * (far - near)) 
				im=np.array(zLinear*255,dtype=np.float64)
				
				depth_im.append(im) # get depth image

				# show_im=Image.fromarray(im)
				# show_im.show()
				idx = np.where(im == np.max(im))
				im[idx] = np.nan

				pcl = intrinsics.deproject(DepthImage(im.astype('float'), frame='view_camera')) # retrieve pointcloud using camera intrinsics 

				pcd = o3d.geometry.PointCloud()
				pcd.points = o3d.utility.Vector3dVector(np.array(pcl.data).T)
				ds_pcd = pcd.voxel_down_sample(0.01)

				class_templates.append(ds_pcd)


			self.templates[class_id] = class_templates
		
			# use images and camera intrinsics to get pointclouds. 


			# do pybullet simulation with random obj orientation. 
				# take picture 
				# pic -> pcd
				# store pcd 


	def get_templates(self, class_id):
		"""
			Return templates related to the given class_id 
		"""

		return self.templates[class_id]

		

	def render_templates(self, class_id): 
		w = self.dataset_config['state_space']['cameras']['general']['im_width']
		h = self.dataset_config['state_space']['cameras']['general']['im_height']
		fx = self.dataset_config['state_space']['cameras']['general']['fx']
		fy = self.dataset_config['state_space']['cameras']['general']['fy']
		cx = self.dataset_config['state_space']['cameras']['general']['cx']
		cy = self.dataset_config['state_space']['cameras']['general']['cy']


		vis = o3d.visualization.Visualizer()

		vis.create_window(width=w, height = h)

		open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()


		open3d_intrinsics.set_intrinsics(width=w, height=h, fx=fx, fy=fy, cx=w/2-0.5, cy=h/2-0.5)

		open3d_camera = o3d.camera.PinholeCameraParameters()
		pose = self.ds_image.get_pose()
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]
		open3d_camera.extrinsic = pose
		open3d_camera.intrinsic = open3d_intrinsics

		view_controls  = vis.get_view_control()
		view_controls.convert_from_pinhole_camera_parameters(open3d_camera)

		pcds = self.templates[class_id]

		for pcd in pcds:
			vis.add_geometry(pcd)

		vis.run()
		vis.destroy_window()

		# shown_pcd = o3d.geometry.PointCloud()
		# shown_pcd.points = o3d.utility.Vector3dVector(np.asarray(templates[0].points))	
		# shown_pcd.paint_uniform_color(np.random.random(3))
		# vis.add_geometry(shown_pcd)

		# for pcd in templates:
		# 	shown_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))	
		# 	vis.update_geometry(shown_pcd)
		# 	vis.poll_events()
		# 	vis.update_renderer()
			#time.sleep(2)

			

			


		#vis.destroy_window()




class MaskedPointclouds():
	"""
		Class encapsulating all operations involved with obtaining pointclouds from the masked areas by sd-maskrcnn. 
	""" 
	def __init__(self):
		self.DatasetImage = None
		self.clusers = None
		self.image = None 


	def feed_image(self, ds_image):
		# reset 
		self.reset()
		# depthImage related to this MaskedPointclouds object 
		self.DatasetImage = ds_image
		self.image = ds_image.get_image()[:,:,0] # depth images have 3 identical channels -> extract first one 
		# number of objects detected within the object
		self.clusters = ds_image.get_nbr_objects_detected()

		self.get_masked_pointclouds()

	def reset(self): 
		self.DatasetImage = None 
		self.clusters = None 
		self.image = None 

		self.pointclouds = None
		self.pointclouds_downsampled = None 
		self.pointclouds_cleaned = None



	def get_total_image_pointcloud(self): 
		"""
			returns open3d pointcloud object of the entire depth image.
		"""
		intrinsics = self.DatasetImage.get_intrinsics()
		pcl_1 = intrinsics.deproject(DepthImage(self.image.astype('float'), frame='view_camera'))
		pcl = np.array(pcl_1.data).T # reshaping 

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(pcl)

		return pcd
	
	def get_masked_pointclouds(self):
		"""
			Method for retrieving masked pointclouds from depth image and masks
			returns nothing but sets the self.pointclouds property
		"""

		intrinsics = self.DatasetImage.get_intrinsics()
		intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}

		pointclouds = []

		all_masks = self.DatasetImage.get_masks()

		for N in range(self.clusters):
			# do calc3points on masks 
			# transform 2d image mask to 3d pointcloud mask 
			mask = all_masks[:,:,N] # extract a mask 
			# im = PIL.Image.fromarray(mask)
			# im.show()
			masked_image = self.image.astype('float') # float type conversion needed if we want to add np.nan
			masked_image[~mask] = np.nan 


			pcl_1 = intrinsics.deproject(DepthImage(masked_image, frame='view_camera')) #pointcloud from depth image 
			pcl = np.array(pcl_1.data).T

			# extract 3d points with none-nan z-value 
			masked_pcl = pcl[~np.isnan(pcl[:,2])] # extract masked part of pointcloud
			
			# store within open3d
			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(masked_pcl)
			pcd.paint_uniform_color(np.random.random(3))

			# way to time consumming for cpu 
			# labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
			# max_label = labels.max()
			# print(f"point cloud has {max_label + 1} clusters")
			# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
			# colors[labels < 0] = 0
			# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
			
			pointclouds.append(pcd)

		self.pointclouds = pointclouds


	def get_pointclouds(self):
		"""
			Returns masked pointclouds
			return type: list of o3d.geometry.PointCloud() objects
		"""
		return self.pointclouds


	def clean_pointclouds(self, downsampled=True): 
		"""
			Masks are often not perfect, resulting in bin pixels being labeled as part of the object. 
			Methods uses clustering techniques to filter out wronly labeld pixels. 	
		"""

		# metrics 'cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’

		print('performing dbscan')
		cleaned_pointclouds = []

		#dbscan = DBSCAN(eps=2, min_samples= 50, metric='euclidean')
		c=0

		if downsampled: 
			pcds = self.pointclouds_downsampled
		else: 
			print('downsampling adviced! ')
			pcds = self.pointclouds

		for pcd in pcds: 
			print('cluster {}/{}'.format(c,self.clusters))
			c+=1
			# points = np.asarray(pcd.points)
			# cluster_results = dbscan.fit(points)
			# # larger cluster is probably what we want 
			# labels = cluster_results.labels_

			# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			# print('{} clusters found'.format(n_clusters_))

			# secondBiggestCluster = Counter(labels).most_common(1)[0][0]
			# unique_labels = np.unique(labels)
			# smallest_label= 10000000
			# for label in unique_labels: 
			# 	if len(np.where(labels==label)[0]) < smallest_label: 
			# 		smallest_label = label 

			#idx = np.where(labels == secondBiggestCluster)[0]
			#Cluster = points[idx]

			# pcd_cluster = o3d.geometry.PointCloud()
			# pcd_cluster.points = o3d.utility.Vector3dVector(points)
			# pcd_cluster.paint_uniform_color(np.random.random(3))
			nbrofpoints = len(np.asarray(pcd.points))
			labels = np.array(pcd.cluster_dbscan(eps=4, min_points=int(nbrofpoints/2), print_progress=False))
			max_label = labels.max()
			print(f"point cloud has {max_label + 1} clusters")
			colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
			colors[labels < 0] = 0
			
			pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

			points = np.asarray(pcd.points)
			new_points = points[np.where(labels!=-1)[0]]

			pcd_cluster = o3d.geometry.PointCloud()
			pcd_cluster.points = o3d.utility.Vector3dVector(new_points)
			




			cleaned_pointclouds.append(pcd_cluster)

		self.pointclouds_cleaned = cleaned_pointclouds


	def downsample_pointclouds(self, voxel_size=0.01): 
		"""
			Downsamples the pointclouds. Necessary for abillity to clean them using dbscan.
		"""
		print('Downsampling:')
		downsampled_pcd = []

		# dbscan = DBSCAN(eps=0.3, min_samples= 20, metric='euclidean')
		c=0
		for pcd in self.pointclouds: 
			print('progress: {}/{}'.format(c, self.clusters))
			c+=1
			ds_pcd = pcd.voxel_down_sample(voxel_size)
			fraction=0.1
			
			#step = int(len(np.asarray(pcd.points))/int(fraction*len(np.asarray(pcd.points))))
			#ds_pcd = pcd.uniform_down_sample(step)
			downsampled_pcd.append(ds_pcd)
			
		
		self.pointclouds_downsampled = downsampled_pcd

	def get_pointcloudsFor_ICP(self): 
		"""
			encapsulates the downsampling and cleaning process of all masked pointclouds. 
		"""

		if self.pointclouds_cleaned != None:
			# assuming self.pointclouds_cleaned already contructed.
			return (self.DatasetImage.get_classes(), self.pointclouds_cleaned)

		else: 
			# in case its not the case. 
			self.downsample_pointclouds()
			self.clean_pointclouds()
			return (self.DatasetImage.get_classes(), self.pointclouds_cleaned)



	def render_masked_pointclouds(self, option=''): 
		"""
			Renders masked pointclouds. 
			options are 'real', 'downsampled' or 'cleaned'
		"""
		intrinsics = self.DatasetImage.get_intrinsics()
		intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}

		vis = o3d.visualization.Visualizer()
		vis.create_window(width=intrin['w'], height = intrin['h'])

		if option == 'cleaned': 
			pcds = self.pointclouds_cleaned
		elif option == 'downsampled': 
			pcds = self.pointclouds_downsampled
		else: 
			pcds = self.pointclouds


		for pcd in pcds:
			vis.add_geometry(pcd)

		
		open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
		open3d_intrinsics.set_intrinsics(width=intrin['w'], height=intrin['h'], fx=intrin['fx'], fy=intrin['fy'], cx=intrin['w']/2-0.5, cy=intrin['h']/2-0.5)

		open3d_camera = o3d.camera.PinholeCameraParameters()
		pose = self.DatasetImage.get_pose()
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]
		open3d_camera.extrinsic = pose
		open3d_camera.intrinsic = open3d_intrinsics

		view_controls  = vis.get_view_control()
		view_controls.convert_from_pinhole_camera_parameters(open3d_camera)


		vis.run()
		vis.destroy_window()


	def render_total_pointcloud(self): 
		"""
			Renders total pointcloud extracted from depth image
		"""
		intrinsics = self.DatasetImage.get_intrinsics()
		intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}

		vis = o3d.visualization.Visualizer()
		vis.create_window(width=intrin['w'], height = intrin['h'])

		pcd = self.get_total_image_pointcloud()
		vis.add_geometry(pcd)
		
		open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
		open3d_intrinsics.set_intrinsics(width=intrin['w'], height=intrin['h'], fx=intrin['fx'], fy=intrin['fy'], cx=intrin['w']/2-0.5, cy=intrin['h']/2-0.5)

		open3d_camera = o3d.camera.PinholeCameraParameters()
		pose = self.DatasetImage.get_pose()
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]
		open3d_camera.extrinsic = pose
		open3d_camera.intrinsic = open3d_intrinsics

		view_controls  = vis.get_view_control()
		view_controls.convert_from_pinhole_camera_parameters(open3d_camera)


		vis.run()
		vis.destroy_window()

	def render_all(self): 
		"""
			Renders masked pointcloud together with total pointcloud
		"""
		intrinsics = self.DatasetImage.get_intrinsics()
		intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}

		vis = o3d.visualization.Visualizer()
		vis.create_window(width=intrin['w'], height = intrin['h'])

		pcd_bin = self.get_total_image_pointcloud()
		print(pcd_bin.points)
		binc = vis.add_geometry(pcd_bin)
		

		for pcd in self.pointclouds:
			vis.add_geometry(pcd)


		open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
		open3d_intrinsics.set_intrinsics(width=intrin['w'], height=intrin['h'], fx=intrin['fx'], fy=intrin['fy'], cx=intrin['w']/2-0.5, cy=intrin['h']/2-0.5)

		open3d_camera = o3d.camera.PinholeCameraParameters()
		pose = self.DatasetImage.get_pose()
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]
		open3d_camera.extrinsic = pose
		open3d_camera.intrinsic = open3d_intrinsics

		view_controls  = vis.get_view_control()
		view_controls.convert_from_pinhole_camera_parameters(open3d_camera)


		vis.run()
		vis.destroy_window()


	def render_templates(self, templates):
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		# target shown
		pcd_bin = self.get_total_image_pointcloud()
		vis.add_geometry(pcd_bin)

		for pcd in self.pointclouds_cleaned:
			for t in templates:
				transformed_temp = getTransformation(pcd, t)
				vis.add_geometry(transformed_temp)
			
		vis.run()
		vis.destroy_window()


	def render_ICP(self, bestTemplates): 
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		# target shown
		pcd_bin = self.get_total_image_pointcloud()
		binc = vis.add_geometry(pcd_bin)
		# real targers are self.pointclouds_cleaned 

		# sources are best templates 
		trf_templates = []
		for (source,target) in zip(bestTemplates, self.pointclouds_cleaned):
			trf_template = getTransformation(target, source)
			trf_templates.append(trf_template)
			vis.add_geometry(trf_template)


		
		threshold = 0.05
		icp_iteration = 200
		save_image = False

		for i in range(icp_iteration):
			for (source, target) in zip(trf_templates, self.pointclouds_cleaned):
				reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold, np.identity(4),
															o3d.pipelines.registration.TransformationEstimationPointToPoint(),
															o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
				source.transform(reg_p2l.transformation)
				vis.update_geometry(source)
			vis.poll_events()
			vis.update_renderer()
		vis.destroy_window()

