import os
import sys
import numpy as np
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


# configs:

FLAGS = parse_args()


# process input:

# mesh to points
extractor = utils.PointsExtractor()
points, normals = extractor.extract(mesh_file_str, depth)

# points to octree
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
  print(np.reshape(y_predict, (-1, 3))*scale + trans)

  
