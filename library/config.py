import numpy

# TODO: rework config system (attribute_dict, yaml, etc.)

K_MAX_ITERATIONS = 10000
K_ROBOT_SPEED = [0.4, 0.0]
K_STEP_TARGET = [0.0, 0.0]

K_SPAWNPOINT_CENTER = 0
K_SPAWNPOINT_UP_LEFT = 0
K_SPAWNPOINT_UP_RIGHT = 0
K_SPAWNPOINT_DOWN_LEFT = 0

K_NEXT_AXIS = [1, 2, 0, 1]
K_FLOAT_EPS = numpy.finfo(numpy.float64).eps

# map axes strings to/from tuples of inner axis, parity, repetition, frame
K_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

K_TUPLE2AXES = dict((v, k) for k, v in K_AXES2TUPLE.items())
K_EPS4 = numpy.finfo(float).eps * 4.0

K_ENVIRONMENT_NAME = 'GazeboEnvironment'
K_LOGGING_PATH = './logs'
K_CHECKPOINT_PATH = './ckpts'

K_NUM_VALID_ACTIONS = 7
K_DOF_SPEED = 2
K_DECAY_RATE = 0.99
K_OBSERVE_TIMESTEPS = 10 
K_EXPLORE_TIMESTEPS = 20000
K_TARGET_EPSILON = 0.0001 
K_INITIAL_EPSILON = 0.1 
K_REPLAY_MEMORY_SIZE = 10000 
K_TARGET_UPDATE_RATE = 0.001 
K_MINIBATCH_SIZE = 8 
K_MAX_EPISODE = 20000
K_MAX_TIMESTEPS = 200

K_DEPTH_IMAGE_WIDTH = 160
K_DEPTH_IMAGE_HEIGHT = 128
K_COLOR_IMAGE_HEIGHT = 228
K_COLOR_IMAGE_WIDTH = 304
K_CHANNEL_COUNT = 3
K_IMAGE_HIST_SIZE = 4
K_H_SIZE = 5120