"""
	code for transformation on depth image. Bases on original code of Olivier. 
"""

import numpy as np 
from .cam_utils import * 
from PIL import Image
from perception import DepthImage
from autolab_core import PointCloud


def depth_transformation(depth, camera, view_camera):

	intrinsics = camera.intrinsics
	view_intrinsics = view_camera.intrinsics

	w = camera.width 
	prev_mat = camera.pose.matrix
	new_mat = view_camera.pose.matrix
	# print(prev_mat)
	# print(new_mat)
	zLinear=depth

	intrin={"fx":intrinsics.fx,"fy":intrinsics.fy,"ppx":intrinsics.cx,"ppy":intrinsics.cy,"w":intrinsics.width,"h":intrinsics.height}	
		
	# far=1.118
	# near=0.458

	# depthSample = 2.0 * depth - 1.0
	# zLinear = 2.0 * near * far / (far + near - depthSample * (far - near))
	# # im=np.array(zLinear*255,dtype=np.uint8)
	# # show_im=Image.fromarray(im)
	# # show_im.show()
	#zLinear = depth
	################################# CAMERA ANGLES ###############################################
	horizontalAngle = 47.5 * np.pi / 180.
	angleResolution=horizontalAngle/w # chosen such that width as array is obtained
	verticalAngle = 36 * np.pi / 180.
	rotationAngleY = -11.75 * np.pi / 180.
	centerPoint=[0.,0.,0.]
	yCoord = np.sin(verticalAngle / 2.)
	xCoord = 0.
	zCoord = np.sqrt(1 - np.sum([np.array([xCoord, yCoord])**2])) #
	triangle = np.append([centerPoint], [[xCoord, yCoord, zCoord], [xCoord, -yCoord, zCoord]], axis = 0)
	transformAngles = np.arange(-horizontalAngle/2, +horizontalAngle/2, angleResolution)
	transTraingles = np.array([TransforY(transformAngle, triangle) for transformAngle in transformAngles])
	unitVecTraingles = np.concatenate([transTraingles[:,1:2,:]-transTraingles[:,:1,:], transTraingles[:,2:3,:]-transTraingles[:,:1,:]], axis = 1)
	unitVecSide1 = unitVecTraingles[:,0,:]
	unitVecSide2 = unitVecTraingles[:,1,:]
	#Calculate the xyz coordinates of the depth map
	pointcloud=calc_3dpoints(zLinear,intrin)
	
	#zwarte zones uitsnijden 
	# Calculating the max x,y point for each image!/CROPPING POINTCLOUD IN YELLOW ZONE
	dist1=zLinear/np.expand_dims(unitVecSide1[:,2],axis=0)
	dist2=zLinear/np.expand_dims(unitVecSide2[:,2],axis=0)
	out1=np.expand_dims(dist1,axis=-1)*np.expand_dims(unitVecSide1,axis=0)
	out2=np.expand_dims(dist2,axis=-1)*np.expand_dims(unitVecSide2,axis=0)
	x1=out1[:,:,0].reshape(-1)
	y1=out1[:,:,1].reshape(-1)
	x2=out2[:,:,0].reshape(-1)
	y2=out2[:,:,1].reshape(-1)
	x_max=np.max(x1)
	x_min=np.min(x1)
	mask=np.logical_and(np.logical_and(np.logical_and(pointcloud[:,1]<y1,pointcloud[:,1]>y2),pointcloud[:,0]>x_min),pointcloud[:,0]<x_max)
	masked_pcd=pointcloud[mask]

	# TRANSFORMING TO PURPLE ZONE
	#schaduw beide kanten
	trans_mat=np.linalg.inv(prev_mat)@np.linalg.inv(new_mat)
	#print(trans_mat)
	trans_mat[0,-1] = -2*trans_mat[0,-1] # fix translation error

	
	masked_transformed_points=transform_points(trans_mat,masked_pcd, True, True)
	transformed_total=transform_points(trans_mat,pointcloud, True, True)

	## PURPLE ZONE POINT CLOUD TO IMAGE
	# points in camera plane of view_Camera 
	img=calc_2points(masked_transformed_points,intrin,devision=1)


	# # transform to center view, camera abovo origin. 
	# prev_mat = new_mat
	# new_mat = np.array([ [ 1. ,     0. ,   0. ,      0. ],
	# 					 [ 0. ,    -1. ,   0. ,      0. ],
	# 					 [ 0. ,     0. ,  -1. ,      0.7215],
	# 					 [ 0. ,     0. ,   0. ,      1.   ]])

	# pointcloud = calc_3dpoints(img, intrin)
	# trans_mat=np.linalg.inv(prev_mat)@np.linalg.inv(new_mat)
	# print(trans_mat)
	# trans_mat[0,-1] = -trans_mat[0,-1]
	# centeredPointCloud = transform_points(trans_mat, pointcloud)

	# img=calc_2points(centeredPointCloud,intrin,devision=1)




	# im=np.array(img*255,dtype=np.uint8)

	# pointcloud = intrinsics.deproject(DepthImage(depth, frame='camera'))
	# pntcld_data = np.array(pointcloud.data).T
	# data = np.c_[pntcld_data, np.ones(len(pntcld_data[:,1]))]
	
	# new_data = trans_mat@data.T
	# #print(new_data.T[:,:3])
	# new_pntcld = PointCloud(new_data.T[:,:3].T, frame='view_camera')
	
	# depth_im = view_intrinsics.project_to_image(new_pntcld)


	return img


