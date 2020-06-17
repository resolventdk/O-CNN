import sys
import numpy as np
import os
import tensorflow as tf 
import json

sys.path.append("..")
import utils




# tensorflow helpers ###########################################################

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:  
        raise ValueError("The input should be numpy ndarray. \
                           Instaed got {}".format(ndarray.dtype))


# vtk2points converter #########################################################

depth = 5  # fixed depth of octree
num_iter = 0  # smoothing iterations
move_lim = 0.0  # node move lim during smoothing iteration
input_path = "dataset"  # 
output_path = "dataset"  # 
mesh_ext = "vtk"  # mesh extension, stl, vtk ,ply...
file_type = 'data'  # name of serialized octree  feature in record

# ordered list of keys
keys = [
    'center.lips.upper.outer',
    'center.nose.tip',
    'left.lips.corner',
    'left.ear.helix.attachement',
    'left.ear.tragus.tip',
    'left.eye.corner_inner',
    'left.eye.corner_outer',
    'right.lips.corner',
    'right.ear.helix.attachement',
    'right.ear.tragus.tip',
    'right.eye.corner_inner',
    'right.eye.corner_outer'
]

octree_flags = {
    "depth":depth,
    "full_depth":2,
    "node_displacement":False,
    "node_feature":False,
    "split_label":False,
    "adaptive":False,
    "adaptive_depth":4,
    "threshold_distance":0.,
    "threshold_normal":0.,
    "key2xyz":False,
    "extrapolate":False,
    "save_pts":False
}

lm_array = np.zeros((len(keys), 3))
lm_feature = _dtype_feature(lm_array)

builder = utils.OctreeBuilder()
extractor = utils.PointsExtractor()

# loop the test train directories of dataset
for directory in ["test", "train"]:

    mesh_files = utils.find_files(input_path + "/" + directory,
                                  "*.%s" % mesh_ext)

    records_prefix = output_path + "/" + directory
    records_name = records_prefix + "_%d_2_1.tfrecords" % depth
    writer = tf.io.TFRecordWriter(records_name)
    
    for index, mesh_file_str in enumerate(mesh_files):

        if ( index % 100 == 0 ):
            print("treating mesh %d / % d" % ( index, len(mesh_files) ) )

        lm_file_str = mesh_file_str.replace(mesh_ext, "json")
        name = mesh_file_str.split("/")[-1].replace(".vtk","")
        octree_name = "%s_%3.3d.octree" % (name, 0)
            
        # landmarks:

        # check if json with landmarks coordinates exists
        if not os.path.exists(lm_file_str):
            print("%s does not exist!" % lm_file_str)
            continue  # next mesh

        # load json
        with open(lm_file_str) as f:
            data = json.load(f)
        landmarks = {}
        for entry in data:
            landmarks[entry['id']]=entry['coordinates']

        # loop landmarks in the predefined order to populate np array
        for i, key in enumerate(keys):
            if key not in landmarks:
                print("%s does not have landmark %s" % (lm_file_str, key))
                continue
            lm_array[i, :] = landmarks[key]

        # mesh to points to octree:

        points, normals = extractor.extract(mesh_file_str, depth)
        
        builder.set_point_cloud("",
                                points.flatten().tolist(),
                                normals.flatten().tolist())
        builder.set_octree_info(octree_flags)
        builder.build_octree()

        # the octree has been centered and normalized so do the same to landmarks    
        lm_array -= builder.center
        lm_array *= 1.0 / builder.radius  # TODO use 2*radius

        # add to tensorflow record
        octree_bytes = builder.octree.get_buffer()
        feature = {file_type: _bytes_feature(octree_bytes),
                   'label': lm_feature(lm_array.flatten()),
                   'index': _int64_feature(index),
                   'filename': _bytes_feature(octree_name.encode('utf8'))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
            
    writer.close()
