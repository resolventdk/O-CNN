import os
import sys
import numpy as np

# TODO: fix incompatible numpy.. supress warning, before loading tf
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.append("../../tensorflow")
sys.path.append("../../tensorflow/script")
from config import parse_args
from network_factory import cls_network
from ocnn import octree_batch

sys.path.append("..")
import utils

# run using
# deploy.py --config configs/deploy_lm_5_2.yaml DEPLOY.input test.vtk


# ordered list of names of landmarks

lm_keys = [
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


# configs:

FLAGS = parse_args()


# process input:

mesh_file_str = FLAGS.DEPLOY.input

assert(mesh_file_str)

# mesh to points
extractor = utils.PointsExtractor()
points, normals = extractor.extract(mesh_file_str, FLAGS.MODEL.depth)

# points to octree

octree_flags = {
    "depth":FLAGS.MODEL.depth,
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

builder = utils.OctreeBuilder()
builder.set_point_cloud("",
                        points.flatten().tolist(),
                        normals.flatten().tolist())
builder.set_octree_info(octree_flags)
builder.build_octree()
octree_bytes = builder.octree.get_buffer()

# scaling parameters
trans = builder.center
scale = builder.radius

# setup network:
    
x = tf.placeholder(tf.string)  # read octree (tf.string)
# octree_batch tf.string -> tf.int8
y = cls_network(octree_batch(x), FLAGS.MODEL, training=False, reuse=False)


# load trained coefficients and process new input:

assert(FLAGS.SOLVER.ckpt)

tf_saver = tf.train.Saver(max_to_keep=10)

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  
  sess.run(tf.global_variables_initializer())
  
  tf_saver.restore(sess, FLAGS.SOLVER.ckpt)

  y_predict = sess.run(y, feed_dict={x: octree_bytes})

  # reshape and rescale
  lm_points = np.reshape(y_predict, (-1, 3))*scale + trans
  
  print("%s predicted landmarks:" % mesh_file_str)
  for i, k in enumerate(lm_keys):
    print("%-50s" % k, lm_points[i,:])

  
