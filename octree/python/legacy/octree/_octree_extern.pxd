# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

# cdef extern from "points.h":
#     cdef cppclass PointsData:
#         int npt
#         const float* pts
#         const float* normals
#         const float* seg
#         const float* features
#         const float* labels

cdef extern from "points_info.h":
    cdef cppclass PointsInfo:
        int pt_num()

# cdef extern from "points.h":
#     cdef cppclass PointsBounds:
#         float radius
#         float center[3]

cdef extern from "points.h" nogil:
    cdef cppclass Points:
        Points()
        bool read_points(const string&)
        bool write_points(const string&)
        bool write_ply(const string&)
        # PointsData get_points_data()
        # PointsBounds get_points_bounds()
        # void center_about(const float*)
        void displace(const float)
        void translate(const float*)
        void rotate(const float, const float*)
        void transform(const float*)
        bool set_points(const vector[float]&,
                        const vector[float]&)
        const float* points()
        const float* normal()        
        const PointsInfo& info()

cdef extern from "octree_info.h" nogil:
    cdef cppclass OctreeInfo:
        OctreeInfo()
        void initialize(int, int, bool,
                        bool, bool, bool, int,
                        float, float, bool,
                        bool, bool, const Points&)  # 3rd and 2nd last are new save_pts, exprapolate
        void set_bbox(float, const float*)
        void set_bbox(const float*, const float*)
        
cdef extern from "octree.h" nogil:
    cdef cppclass Octree:
        Octree()
        void build(const OctreeInfo&, const Points&);
        bool write_octree(const string&)
        string get_binary_string()
        void octree2mesh(vector[float]&, vector[int]&, int, int)
        OctreeInfo& mutable_info()
        
cdef extern from "math_functions.h" nogil:
    void bounding_sphere(float&, float*, const float*, const int);        
