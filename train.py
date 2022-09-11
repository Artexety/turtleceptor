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
logger = logging.getLogger('trainer')


def update_target_graph(variables, tau) -> list:
    total_vars = len(variables)
    operations = []	
    for i, v in enumerate(variables[0:total_vars // 2]):
        operations.append(
            variables[i + total_vars // 2].assign((v.value() * tau) + ((1 - tau) * variables[i + total_vars // 2].value()))
        )
    return operations


def update_target_network(operations, session) -> None:
	for o in operations:
		session.run(o)


def train_network():
    session = tf.InteractiveSession()
    
    with tf.name_scope("online_network"):
        online_network = QNetwork(session)
    
    with tf.name_scope("target_network"):
        target_network = QNetwork(session)
    
    rospy.sleep(1.0)

    reward_var = tf.Variable(0.0, trainable=False)
    reward_epi = tf.summary.scalar('reward', reward_var)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(K_LOGGING_PATH, session.graph)

    # initialize gazebo environment (./worlds/train_world.world)
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
    
    internal_buffer = deque()
    depth_data_init = environ.get_depth_observation()
    depth_data_memory = np.stack((depth_data_init, depth_data_init, depth_data_init, depth_data_init), axis=2)
    terminal = False

    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables)
    session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(K_CHECKPOINT_PATH)
    logger.info('found checkpoint: \n\n' + str(checkpoint))
    
    if checkpoint and checkpoint.model_checkpoint_path:
        trainable_saver.restore(session, checkpoint.model_checkpoint_path)
        logger.info('successfully loaded network from: ' + checkpoint.model_checkpoint_path)
    else:
        logger.info('could not find any old network weights!')
    
    # start with the training
    current_episode = 0
    epsilon = K_INITIAL_EPSILON
    
    r_epi = 0.0
    step = 0
    T = 0
    
    rate = rospy.Rate(5)
    logger.info('num_trainable_variables: ' + str(len(trainables)))
    
    operations = update_target_graph(trainables, K_TARGET_UPDATE_RATE)
    loop_time = time.time()
    last_loop_time = loop_time
    
    while current_episode < K_MAX_EPISODE and not rospy.is_shutdown():
        environ.reset_environment()
        
        r_epi = 0.0
        step = 0
        
        terminal = False
        do_reset = False
        loop_time_buffer = []
        action_index = 0
        
        while not do_reset and not rospy.is_shutdown():
            depth_data_current = environ.get_depth_observation()
            depth_data_current = np.reshape(depth_data_current, (K_DEPTH_IMAGE_HEIGHT, K_DEPTH_IMAGE_WIDTH, 1))
            depth_data_memory = np.append(depth_data_current, depth_data_memory[:, :, :(K_IMAGE_HIST_SIZE - 1)], axis=2)
            
            reward_t, terminal, do_reset = environ.calculate_reward(step)
            if step > 0:
                internal_buffer.append((depth_data_memory, a_t, reward_t, depth_data_memory, terminal))
                if len(internal_buffer) > K_REPLAY_MEMORY_SIZE:
                    internal_buffer.popleft()

            depth_data_memory = depth_data_memory
            a = session.run(online_network.readout, feed_dict = {online_network.state : [depth_data_memory]})
            a_t = np.zeros([K_NUM_VALID_ACTIONS])
            r_t = a[0]
            
            if current_episode <= K_OBSERVE_TIMESTEPS:
                action_index = random.randrange(K_NUM_VALID_ACTIONS)
                a_t[action_index] = 1
            else:
                if random.random() <= epsilon:
                    action_index = random.randrange(K_NUM_VALID_ACTIONS)
                    a_t[action_index] = 1
                    logger.info('forcing random action --> action_index: ' + str(action_index))
                else:
                    action_index = np.argmax(r_t)
                    a_t[action_index] = 1

            environ.publish_controller_input(action_index)

            if current_episode > K_OBSERVE_TIMESTEPS:
                minibatch = random.sample(internal_buffer, K_MINIBATCH_SIZE)
                y_batch = []
                depth_memory_t0_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                depth_memory_t1_batch = [d[3] for d in minibatch]
                
                q0_value = online_network.readout.eval(feed_dict = {online_network.state : depth_memory_t1_batch})
                q1_valua = online_network.readout.eval(feed_dict = {online_network.state : depth_memory_t1_batch})
                
                for i in range(0, len(minibatch), 1):
                    terminal_batch = minibatch[i][4]
                    if terminal_batch:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + K_DECAY_RATE * q1_valua[i, np.argmax(q0_value[i])])

                # update target network with the target values generated by the online network
                online_network.train_step.run(
                    feed_dict={
                        online_network.y : y_batch,
                        online_network.a : a_batch,
                        online_network.state : depth_memory_t0_batch 
                    }
                )
                update_target_network(operations, session)

            if epsilon > K_TARGET_EPSILON and current_episode > K_OBSERVE_TIMESTEPS:
                epsilon -= (K_INITIAL_EPSILON - K_TARGET_EPSILON) / K_EXPLORE_TIMESTEPS

            r_epi += reward_t
            step += 1
            T += 1
            
            last_loop_time = loop_time
            loop_time = time.time()
            loop_time_buffer.append(loop_time - last_loop_time)
            rate.sleep()

        if current_episode > K_OBSERVE_TIMESTEPS:
            summary_string = session.run(merged_summary, feed_dict={reward_var: r_epi})
            summary_writer.add_summary(summary_string, current_episode - K_OBSERVE_TIMESTEPS)

        if (current_episode + 1) % 50 == 0 :
            trainable_saver.save(session, K_CHECKPOINT_PATH + '/DQN-' + K_ENVIRONMENT_NAME, global_step = current_episode)

        if len(loop_time_buffer) == 0:
            logger.info('current_episode: ' + str(current_episode) + ' | current_reward: ' + str(r_epi) + ' | T: ' + str(T))
        else:
            logger.info('current_episode: ' + str(current_episode) + ' | current_reward: ' + str(r_epi) + ' | T: ' + str(T) + ' | t: ' + str(np.mean(loop_time_buffer)))

        current_episode += 1


if __name__ == '__main__':
    logger.info("hello world from logging")
    train_network()    

