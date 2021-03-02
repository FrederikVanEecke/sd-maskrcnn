"""
	functionalities facilitating the usage and manipulation of point clouds. both ground thruth point cloud as masked poinclouds

	Author: Frederik Van Eecke
"""

import os
import sys
sys.path.append("/home/frederik/Documents/GitHub/sd-maskrcnn")
from .cam_utils import * 

import open3d as o3d
import PIL
import cv2
from perception import DepthImage
from sklearn.cluster import DBSCAN 
from collections import Counter
import matplotlib.pyplot as plt 
import time

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

		self.meshes_dir = self.dataset_config['urdf_cache_dir'] 
		path = self.pose_config['pointcloud_templates']['path']

		if not os.path.isabs(path): 
			path = os.path.join(os.getcwd(), path)

		self.templates_dir = os.path.join(path, 'pointcloud_templates')

		if not os.path.exists(self.templates_dir): 
			os.mkdir(self.templates_dir)

		self.camera_pose =  np.array([[ 1.  ,        0.   ,       0.    ,      0.        ],
							 [ 0.    ,     -1.    ,      0.    ,      0.        ],
							 [ 0.    ,      0.    ,     -1.    ,      0.72149998],
							 [ 0.    ,      0.    ,      0.    ,      1.        ]])


		#self.mapping = {1:'ycb~cubei7'} # temporary mapping for 1 cube case. 
		self.mapping = {1: 'ycb~monkeyHead'}


	def get_templates(self, class_id): 
		# check if template folder exists for this class.
		class_name  = self.mapping[class_id] 
		path = os.path.join(self.templates_dir, class_name)
		if not os.path.exists(path): 
			# create folder 
			os.mkdir(path)
			# create pointcloud templates within this folder 
			self.create_templates(path, class_name)

	
		templates = self.read_templates(path)
		return templates 


	def create_templates(self, templates_path, class_name): 
		"""
			Creates pointcloud templates of the class class_name within the given  path. 
		"""

		nbrOfTemplates = self.pose_config['pointcloud_templates']['viewpoints']

		# read obj file from meshes dir 
		rel_path_to_obj = os.path.join(self.meshes_dir, '{}/{}_convex_piece_0.obj'.format(class_name, class_name))
		abs_path_to_obj = os.path.join(self.pose_config['dir']['path'], rel_path_to_obj)

		obj_mesh = o3d.io.read_triangle_mesh(abs_path_to_obj) 

		pcd = obj_mesh.sample_points_uniformly(number_of_points=10000, use_triangle_normal=False, seed=- 1)
		#pcd.translate(translate=np.array([0,0,0]), relative=False) # translate to the origin. 
		#pcd.scale(3,np.zeros(3))

		points = np.asarray(pcd.points)
		
		o3d.io.write_point_cloud(os.path.join(templates_path, '{}.pcd'.format(class_name)), pcd)

		for idx in range(nbrOfTemplates): 
			# random rotation matrix 
			rot=R.random(random_state=np.random.randint(1000)).as_matrix()
			rotatedPoints = points@rot 

			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(rotatedPoints)
			pcd.paint_uniform_color(np.random.random(3))

			o3d.io.write_point_cloud(os.path.join(templates_path, '{}_rot{}.pcd'.format(class_name, idx)), pcd)

		return None 

	def read_templates(self, path): 
		"""
			read and return all template pointclouds within the given path.
			returns list with open3d pointcloud objects 
		"""
		templates = [ o3d.io.read_point_cloud(os.path.join(path, pcdfile)) for pcdfile in os.listdir(path) ]


		return templates

	def set_camera_pose(self, pose): 
		self.camera_pose = pose

	def render_templates(self, class_id): 
		templates = self.get_templates(class_id)
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
		pose = self.camera_pose
		pose[2,3] = -pose[2,3] # convention mismatches (i guess)
		pose[1,1] = -pose[1,1]
		pose[2,2] = -pose[2,2]
		open3d_camera.extrinsic = self.camera_pose
		open3d_camera.intrinsic = open3d_intrinsics

		view_controls  = vis.get_view_control()
		view_controls.convert_from_pinhole_camera_parameters(open3d_camera)

		for pcd in templates:
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
			print(labels)
			pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

			points = np.asarray(pcd.points)
			new_points = points[np.where(labels!=-1)[0]]

			pcd_cluster = o3d.geometry.PointCloud()
			pcd_cluster.points = o3d.utility.Vector3dVector(new_points)
			




			cleaned_pointclouds.append(pcd)

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
			print(pcd)
			#step = int(len(np.asarray(pcd.points))/int(fraction*len(np.asarray(pcd.points))))
			#ds_pcd = pcd.uniform_down_sample(step)
			downsampled_pcd.append(ds_pcd)
			print(ds_pcd)
		
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

