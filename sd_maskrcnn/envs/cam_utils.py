# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:02:16 2020

@author: Olivier
"""
import numpy as np
import cv2
import open3d as o3d
import os

def rotx(angle):
    """ angle in radians"""
    """ returns 4x4 tranformation matrix """
    return np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],[0,np.sin(angle),np.cos(angle),0],[0,0,0,1]])
def roty(angle):
    """ angle in radians"""
    """ returns 4x4 tranformation matrix """
    return np.array([[np.cos(angle),0,np.sin(angle),0],[0,1,0,0],[-np.sin(angle),0,np.cos(angle),0],[0,0,0,1]])  
  
def rotz(angle):
    """ angle in radians"""
    """ returns 4x4 tranformation matrix """
    return np.array([[np.cos(angle),-np.sin(angle),0,0],[np.sin(angle),np.cos(angle),0,0],[0,0,1,0],[0,0,0,1]])

def trans(dist):
    a=np.eye(4)
    a[:3,3]=dist
    return a

def calc_3dpoints(depth,intrin):
    h,w=depth.shape
    a,b=np.meshgrid(np.array(range(h)),np.array(range(w)))
    total=np.c_[b.T.reshape(-1),a.T.reshape(-1),depth.reshape(-1)] 
    total[:,0]=(total[:,0]-intrin["ppx"])*total[:,2]/intrin["fx"]
    total[:,1]=(total[:,1]-intrin["ppy"])*total[:,2]/intrin["fy"]
    return total

def calc_2points(points,intrin,devision=1,image=True):
    pixel_x=np.array(np.clip(np.round(intrin["ppx"]/devision+(intrin["fx"]/devision)*points[:,0]/points[:,2]),0,int(intrin["w"]/devision)-1),dtype=np.int32)
    pixel_y=np.array(np.clip(np.round(intrin["ppy"]/devision+(intrin["fy"]/devision)*points[:,1]/points[:,2]),0,int(intrin["h"]/devision)-1),dtype=np.int32)
    
    if image:
        new_shape=(int(intrin["h"]/devision),int(intrin["w"]/devision))
        img=np.zeros(new_shape)
        idxs_sort=np.argsort(points[:,2])[::-1] 
        img[pixel_y[idxs_sort],pixel_x[idxs_sort]]=points[idxs_sort,2]
        return img
    else:
        return pixel_x,pixel_y
  
def transform_points(mat,points,change_axis_start=True,change_axis_end=True):
    """ Assuming the matrix is in the opengl framework and we need the opencv framework matrix """
    new_points=points.copy()
    if change_axis_start:
        #new_points[:,1]=-new_points[:,1]
        new_points[:,2]=-new_points[:,2]
    new_points=np.c_[new_points,np.ones(len(new_points))]
    transformed_points=mat@new_points.T
    transformed_points=transformed_points.T[:,:3]
    if change_axis_end:
        #transformed_points[:,1]=-transformed_points[:,1]
        transformed_points[:,2]=-transformed_points[:,2]
    return transformed_points
    
def TransforY(rotationAngleY, Points):
    """ Transforms Points in Y direction """
    cRoty = np.cos(rotationAngleY)
    sRoty = np.sin(rotationAngleY)
    transMatrix = np.array(((cRoty,0.,sRoty),(0.,1.,0.),(-sRoty,0.,cRoty)))
    
    return np.dot(transMatrix,np.transpose(Points)).transpose()

def load_obj(file_name,obj_points):
    path=os.path.join("Simulation_files",file_name)
    mesh=o3d.io.read_triangle_mesh(path)
    pcd=mesh.sample_points_poisson_disk(number_of_points=obj_points)
    points=np.asarray(pcd.points)/1000.
    return points





