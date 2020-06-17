"""Tools for building octree"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from legacy.octree._octree import Octree
from legacy.octree._octree import OctreeInfo
from legacy.octree._octree import Points

class OctreeBuilder:
    
    def set_point_cloud(self, file_path, points, normals):

        # load point cloud
        self.point_cloud = Points(file_path, points, normals)
        
        # TODO: deal with empty points
        
        # bounding sphere
        self.radius, self.center = self.point_cloud.get_points_bounds()
        # print(self.radius, self.center)
        
        # centralize & displacement
        self.point_cloud.translate(-self.center)
        # TODO: displacement
        
    def set_octree_info(self, flags):

        self.octree_info = OctreeInfo()
        self.octree_info.initialize(
            flags["depth"],
            flags["full_depth"],
            flags["node_displacement"],
            flags["node_feature"],
            flags["split_label"],
            flags["adaptive"],
            flags["adaptive_depth"],
            flags["threshold_distance"],
            flags["threshold_normal"],
            flags["key2xyz"],
            flags["extrapolate"],
            flags["save_pts"],
        self.point_cloud)
        
        # the point cloud has been centralized in "set_point_cloud"
        # so initialize the bbmin & bbmax in the following way
        bbox_min = - self.radius * np.ones(3, dtype=np.single)
        bbox_max = self.radius * np.ones(3, dtype=np.single)
        self.octree_info.set_bbox1(bbox_min, bbox_max)        
        
    def build_octree(self):
        # when building octree, points will be scaled as
        # mul = 1. / bbox_max_width
        # pts_scaled = (pts - bbmin) * mul;
        self.octree = Octree(self.octree_info, self.point_cloud)
        
        # Modify the bounding box before saving so it reflects
        # the unscaled, uncentralized point cloud
        self.octree.set_bbox(self.radius, self.center)
                
    def save_octree(self, output_filename):
        self.octree.write_file(output_filename)
    

    
