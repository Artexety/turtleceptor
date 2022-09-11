from __future__ import print_function
from library.environment import GazeboEnvironment
import sys, os, warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

# remove ros python installation from python path to prevent name 
# mismatches (ros-distro: kinetic, ubuntu 16.04) 
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import random
import time

from collections import deque
from library.network import *
from library.config import *

import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('checker')


def validate_network():
    session = tf.InteractiveSession()
    
    with tf.device("/cpu:0"):
        with tf.name_scope("online_network"):
            online_net = QNetwork(session)
        with tf.name_scope("target_network"):
            target_net = QNetwork(session)
    rospy.sleep(1.0)


    # initialize gazebo environment (./worlds/test_world.world)
    # real world environment -> get depth through our edge_depth netowrk
    environ = GazeboEnvironment()
    logger.info('gazebo_environment was successfully initialized')
    
    q_network_params = []
    
    for variable in tf.trainable_variables():
        variable_name = variable.name
        if variable_name.find('online_network') >= 0:
            q_network_params.append(variable)
    
    for variable in tf.trainable_variables():
        variable_name = variable.name
        if variable_name.find('target_network') >= 0:
            q_network_params.append(variable)
    
    logger.info('printing q-network variables: \n')
    for i, v in enumerate(q_network_params):
        print('  var {:02d}: {:15}   {}'.format(i, str(v.get_shape()), v.name))
    print()
    
    q_network_saver = tf.train.Saver(q_network_params, max_to_keep=1)

    checkpoint = tf.train.get_checkpoint_state(K_CHECKPOINT_PATH)
    logger.info('found checkpoint: \n\n' + str(checkpoint))
    if checkpoint and checkpoint.model_checkpoint_path:
        q_network_saver.restore(session, checkpoint.model_checkpoint_path)
        logger.info('successfully loaded q-network from: ' + checkpoint.model_checkpoint_path)
    else:
        logger.critical('could not find any q-network weights!')
    
    rate = rospy.Rate(5)
    
    while not rospy.is_shutdown():
        environ.reset_environment()

        action_index = 0
        is_collided = False
        depth_data_init = environ.get_depth_observation()
        depth_data_memory = np.stack((depth_data_init, depth_data_init, depth_data_init, depth_data_init), axis=2)
        
        while not is_collided:
            if environ.simulation_params.bump:
                is_collided = True
                continue
             
            depth_data_current = environ.get_depth_observation()
            depth_data_current = np.reshape(depth_data_current, (K_DEPTH_IMAGE_HEIGHT, K_DEPTH_IMAGE_WIDTH, 1))
            depth_data_memory = np.append(depth_data_current, depth_data_memory[:, :, :(K_IMAGE_HIST_SIZE - 1)], axis=2)

            # choose an action greedily
            a = session.run(online_net.readout, feed_dict = {online_net.state : [depth_data_memory]})
            a_t = np.zeros([K_NUM_VALID_ACTIONS])
            r_t = a[0]
            
            action_index = np.argmax(r_t)
            a_t[action_index] = 1
            logger.info('turtlebot performed action --> action_index: ' + str(action_index))

            environ.publish_controller_input(action_index)
            rate.sleep()


if __name__ == '__main__':
    logger.info("hello world from logging")
    validate_network()    