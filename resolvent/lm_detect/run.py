import os
import sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

sys.path.append("../../tensorflow")
sys.path.append("../../tensorflow/script")
from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_factory import cls_network
from ocnn import loss_functions_regress

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# configs
FLAGS = parse_args()

#define the graph
def compute_graph(dataset='train', training=True, reuse=False):
  flags_data = FLAGS.DATA.train if dataset=='train' else FLAGS.DATA.test
  octree, label = DatasetFactory(flags_data)()

  logit = cls_network(octree, FLAGS.MODEL, training, reuse)  # reuse cls network, switch loss function
  losses = loss_functions_regress(logit, label,
                                  FLAGS.LOSS.weight_decay, 'ocnn', FLAGS.LOSS.label_smoothing)
  losses.append(losses[0] + losses[2]) # total loss
  names  = ['loss', 'accu', 'regularizer', 'total_loss']
  return losses, names

# run the experiments
if __name__ == '__main__':
  solver = TFSolver(FLAGS.SOLVER, compute_graph)
  solver.run()

# flags_data = FLAGS.DATA.train
# octree, label = DatasetFactory(flags_data)()
