from legacy.octree cimport _octree_extern
#from legacy.dataset._writable_data cimport WritableData

import numpy as np
cimport numpy as np

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

def _check_array(array, shape):
    if shape != array.shape:
        raise ValueError('Illegal array dimensionality {0}, expected {1}'.format(array.shape, shape))

def _ensure_contiguous(np.ndarray array):
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    return array

cdef class Points:
    cdef _octree_extern.Points *c_points  # pointer to a c++ Points object

    # unlike in c++ etc your can only have 1 constructor in python
    def __cinit__(self, filename="", points=[], normals=[]):
        cdef string stl_string = filename.encode('UTF-8')
        cdef bool points_read
        cdef bool points_set
        cdef vector[float] points_cppvec = points
        cdef vector[float] normals_cppvec = normals
        if filename:  # init from file
            print(filename, stl_string)
            with nogil:
                self.c_points = new _octree_extern.Points()
                points_read = self.c_points.read_points(stl_string)
            if not points_read:
                raise RuntimeError('Could not read Points file: {0}'.format(filename))
        else:  # init from list of points, and normals
            # TODO: check consistent arguments
            with nogil:
                self.c_points = new _octree_extern.Points()
                points_set = self.c_points.set_points(points_cppvec, normals_cppvec)
            if not points_set:
                raise RuntimeError('Could not initialize Points from lists')
        
    def __dealloc__(self):
        del self.c_points  # now that c_point is a pointer we have to dealloc
        
    def write_file(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        with nogil:
            self.c_points.write_points(stl_string)

    def center(self):
        _, center = self.get_points_bounds()
        self.center_about(center)

    def center_about(self, np.ndarray center):
        center = _ensure_contiguous(center)
        _check_array(center, (3,))

        trans = -center
        cdef float[::1] trans_view = trans.ravel()

        with nogil:
            self.c_points.translate(&trans_view[0])

    def translate(self, np.ndarray translate):
        cdef float[::1] trans_view = translate.ravel()

        with nogil:
            self.c_points.translate(&trans_view[0])
        
            
    def displace(self, float displacement):
        with nogil:
            self.c_points.displace(displacement)

    def rotate(self, float angle, np.ndarray axis):
        axis = _ensure_contiguous(axis)
        _check_array(axis, (3,))

        cdef float[::1] axis_view = axis.ravel()
        with nogil:
            self.c_points.rotate(angle, &axis_view[0])

    def transform(self, np.ndarray transformation_matrix):
        transformation_matrix = _ensure_contiguous(transformation_matrix)
        _check_array(transformation_matrix, (3,3))

        cdef float[::1] mat_view = transformation_matrix.ravel()

        with nogil:
            self.c_points.transform(&mat_view[0])

    def get_points_bounds(self):
        cdef _octree_extern.PointsInfo info = self.c_points.info()
        cdef float radius = 0.0
        cdef float[::1] center_view = np.zeros(3, dtype=np.single).ravel()
        # with nogil:
        _octree_extern.bounding_sphere(radius,
                                       &center_view[0],
                                       self.c_points.points(),
                                       info.pt_num())

        center = np.empty_like (center_view)
        center[:] = center_view

        return radius, center
    
    def get_points_data(self):
        cdef _octree_extern.PointsInfo info = self.c_points.info()
        cdef Py_ssize_t nrows = info.pt_num(), ncols=3
        cdef float[:,::1] points_view = <float[:nrows, :ncols]> self.c_points.points()
        cdef float[:,::1] normals_view = <float[:nrows, :ncols]> self.c_points.normal()

        points = np.empty_like (points_view)
        points[:] = points_view
        normals = np.empty_like (normals_view)
        normals[:] = normals_view

        return points, normals

    def write_ply(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        with nogil:
            self.c_points.write_ply(stl_string)

    
cdef class OctreeInfo:
    cdef _octree_extern.OctreeInfo c_octree_info
    def initialize(
            self,
            int depth,
            int full_depth,
            bool node_displacement,
            bool node_feature,
            bool split_label,
            bool adaptive,
            int adaptive_depth,
            float threshold_distance,
            float threshold_normal,
            bool key2xyz,
            bool save_pts,
            bool extrapolate,
            Points points):
        c_points_ptr = points.c_points
        self.c_octree_info.initialize(
                depth,
                full_depth,
                node_displacement,
                node_feature,
                split_label,
                adaptive,
                adaptive_depth,
                threshold_distance,
                threshold_normal,
                key2xyz,
                save_pts,
                extrapolate,
                dereference(c_points_ptr))

    def set_bbox(self, float radius, np.ndarray center):
        center = _ensure_contiguous(center)
        _check_array(center, (3,))
        cdef float[::1] center_view = center.ravel()

        with nogil:
            self.c_octree_info.set_bbox(radius, &center_view[0])

    def set_bbox1(self, np.ndarray bbox_min, np.ndarray bbox_max):

        bbox_min = _ensure_contiguous(bbox_min)
        _check_array(bbox_min, (3,))
        cdef float[::1] bbox_min_view = bbox_min.ravel()

        bbox_max = _ensure_contiguous(bbox_max)
        _check_array(bbox_max, (3,))
        cdef float[::1] bbox_max_view = bbox_max.ravel()

        with nogil:
            self.c_octree_info.set_bbox(&bbox_min_view[0], &bbox_max_view[0])
            
# cdef class Octree(WritableData):
            #self.cpp_string = self.c_octree.get_binary_string()
cdef class Octree():
    cdef _octree_extern.Octree c_octree

    def __cinit__(self, OctreeInfo info, Points points):
        c_points_ptr = points.c_points
        c_info_ptr = &info.c_octree_info
        with nogil:
            self.c_octree.build(dereference(c_info_ptr), dereference(c_points_ptr))

    def get_buffer(self):
        cdef string cpp_string
        with nogil:
            cpp_string = self.c_octree.get_binary_string()
        return cpp_string
            
    def set_bbox(self, float radius, np.ndarray center):
        center = _ensure_contiguous(center)
        _check_array(center, (3,))
        cdef float[::1] center_view = center.ravel()

        with nogil:
            self.c_octree.mutable_info().set_bbox(radius, &center_view[0])
            
    def write_file(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        with nogil:
            self.c_octree.write_octree(stl_string)

            
    # def write_obj(self, filename, int depth_start, int depth_end):
    #     cdef vector[float] V
    #     cdef vector[int] F
    #     self.c_octree.octree2mesh(V, F, depth_start, depth_end)
    #     _octree_extern.write_obj(filename, V, F)
