import rospy
import sys, os, warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent

# remove ros python installation from python path to prevent name 
# mismatches (ros-distro: kinetic, ubuntu 16.04) 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import random
import numpy
import time
import copy
import cv2

from .transforms import convert_quaternion_to_euler, convert_euler_to_quaternion
from .collections import AttributeDict
from .config import *

import logging
import logging.config

logging.config.fileConfig(os.path.join(os.path.abspath(__file__).split('/library')[0], 'logging.conf'))
logger = logging.getLogger('environ')


class GazeboEnvironment(object):

    def __init__(self) -> None:
        rospy.init_node(K_ENVIRONMENT_NAME, anonymous=False)

        self.default_robot_state = ModelState()
        self.default_robot_state.model_name = 'mobile_base' 
        self.default_robot_state.pose.position.x = 0.5
        self.default_robot_state.pose.position.y = 0.0
        self.default_robot_state.pose.position.z = 0.0
        self.default_robot_state.pose.orientation.x = 0.0
        self.default_robot_state.pose.orientation.y = 0.0
        self.default_robot_state.pose.orientation.z = 0.0
        self.default_robot_state.pose.orientation.w = 1.0
        self.default_robot_state.twist.linear.x = 0.0
        self.default_robot_state.twist.linear.y = 0.0
        self.default_robot_state.twist.linear.z = 0.0
        self.default_robot_state.twist.angular.x = 0.0
        self.default_robot_state.twist.angular.y = 0.0
        self.default_robot_state.twist.angular.z = 0.0
        self.default_robot_state.reference_frame = 'world'

        self.simulation_params = AttributeDict()
        self.simulation_params.depth_image_size = [K_DEPTH_IMAGE_WIDTH, K_DEPTH_IMAGE_HEIGHT]
        self.simulation_params.color_image_size = [K_COLOR_IMAGE_WIDTH, K_COLOR_IMAGE_HEIGHT]
        self.simulation_params.vision_bridge = CvBridge()

        self.simulation_params.current_robot_state = None
        self.simulation_params.object_state = [0, 0, 0, 0]
        self.simulation_params.object_label = []

        # 0. | left 90/s | left 45/s | right 45/s | right 90/s | acc 1/s | slow down -1/s
        self.simulation_params.action_table = [0.4, 0.2, numpy.pi / 6, numpy.pi / 12, 
                                               0.0, -numpy.pi / 12, -numpy.pi / 6]

        self.simulation_params.robot_speed = K_ROBOT_SPEED
        self.simulation_params.default_states = None
        self.simulation_params.start_time = time.time()
        self.simulation_params.max_iterations = K_MAX_ITERATIONS
        self.simulation_params.depth_image = None
        self.simulation_params.color_image = None
        self.simulation_params.scan = None
        self.simulation_params.bump = False

        self.simulation_params.odometry = AttributeDict()
        self.simulation_params.odometry.linear_x_speed = 0
        self.simulation_params.odometry.linear_y_speed = 0
        self.simulation_params.odometry.rotation_speed = 0

        self.ros_publisher = AttributeDict()
        self.ros_publisher.cmd_velocity = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)
        self.ros_publisher.update_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
        self.ros_publisher.resized_depth = rospy.Publisher('camera/depth/image_resized',Image, queue_size = 10)
        self.ros_publisher.resized_color = rospy.Publisher('camera/rgb/image_resized',Image, queue_size = 10)

        self.ros_subscriber = AttributeDict()
        self.ros_subscriber.object_state = rospy.Subscriber('gazebo/model_states', ModelStates, self.model_state_callback)
        self.ros_subscriber.color_image = rospy.Subscriber('camera/rgb/image_raw', Image, self.color_image_callback)
        self.ros_subscriber.depth_image = rospy.Subscriber('camera/depth/image_raw', Image, self.depth_image_callback)
        self.ros_subscriber.bumper = rospy.Subscriber('mobile_base/events/bumper', BumperEvent, self.bumper_callback)
        self.ros_subscriber.laserscan = rospy.Subscriber('scan', LaserScan, self.laserscan_callback)
        self.ros_subscriber.odom = rospy.Subscriber('odom', Odometry, self.odometry_callback)

        rospy.sleep(2)
        rospy.on_shutdown(self.__shutdown__)
    
    def __shutdown__(self) -> None:
        logger.info('Triggered system shutdown. This can take up to several seconds!')
        self.ros_publisher.cmd_velocity.publish(Twist())
        rospy.sleep(1)
    
    def model_state_callback(self, data) -> None:
        i = data.name.index("mobile_base")	
        quaternion = [
            data.pose[i].orientation.x,
			data.pose[i].orientation.y,
			data.pose[i].orientation.z,
			data.pose[i].orientation.w
        ]

        angle = convert_quaternion_to_euler(quaternion)
        self_current_yaw = angle[2]
        self.simulation_params.current_robot_state = [
            data.pose[i].position.x,
            data.pose[i].position.y,
            self_current_yaw,
            data.twist[i].linear.x,
			data.twist[i].linear.y,
			data.twist[i].angular.z
        ]

        for j in range (0, len(self.simulation_params.object_label), 1):
            i = data.name.index(self.self.simulation_params.object_label[j])
            quaternion = [
                data.pose[i].orientation.x,
                data.pose[i].orientation.y,
                data.pose[i].orientation.z,
                data.pose[i].orientation.w
            ]

            angle = convert_quaternion_to_euler(quaternion)
            other_current_yaw = angle[2]
            self.simulation_params.object_state[j] = [
                data.pose[i].position.x, 
				data.pose[i].position.y,
				other_current_yaw
            ]
        
        if self.simulation_params.default_states is None:
            self.simulation_params.default_states = copy.deepcopy(data)
    
    def depth_image_callback(self, image_data) -> None:
        self.simulation_params.depth_image = image_data
    
    def color_image_callback(self, image_data) -> None:
        self.simulation_params.color_image = image_data
    
    def laserscan_callback(self, scan_data) -> None:
        self.simulation_params.scan = numpy.array(scan_data.ranges)
    
    def odometry_callback(self, odometry_data) -> None:
        self.simulation_params.odometry.linear_x_speed = odometry_data.twist.twist.linear.x
        self.simulation_params.odometry.linear_y_speed = odometry_data.twist.twist.linear.y
        self.simulation_params.odometry.rotation_speed = odometry_data.twist.twist.angular.z

    def bumper_callback(self, bumper_data) -> None:
        if bumper_data.state == BumperEvent.PRESSED:
            self.simulation_params.bump = True
        else:
            self.simulation_params.bump = False
    
    def get_depth_observation(self, simulate_noise = False) -> numpy.ndarray:
        image_data = numpy.frombuffer(self.simulation_params.depth_image.data, dtype = numpy.float32).reshape(
            self.simulation_params.depth_image.height, self.simulation_params.depth_image.width, 1)

        dsize = (self.simulation_params.depth_image_size[0], self.simulation_params.depth_image_size[1])
        image_data = cv2.resize(image_data, dsize, interpolation = cv2.INTER_AREA)

        image_data[numpy.isnan(image_data)] = 0.0
        image_data[image_data < 0.42] = 0.0

        if simulate_noise:
            image_data /= (10.0 / 255.0)
            gauss_mask = numpy.random.normal(0.0, 0.5, dsize)
            gauss_mask = gauss_mask.reshape(dsize[1], dsize[0])
            
            image_data = numpy.array(image_data, dtype=numpy.float32)
            image_data = image_data + gauss_mask
            image_data[image_data < 0.00001] = 0.0
            image_data *= (10.0 / 255.0)

        try:
            resized_image = self.simulation_params.vision_bridge.cv2_to_imgmsg(image_data, "passthrough")
        except Exception as e:
            raise e
        self.ros_publisher.resized_depth.publish(resized_image)
        return (image_data / 5.0)
    
    def get_color_observation(self) -> numpy.ndarray:
        image_data = numpy.frombuffer(self.simulation_params.color_image.data, dtype = numpy.uint8).reshape(
            self.simulation_params.color_image.height, self.simulation_params.color_image.width, -1)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
		
        dsize = (self.simulation_params.color_image_size[0], self.simulation_params.color_image_size[1])
        cv_resized_image = cv2.resize(image_data, dsize, interpolation = cv2.INTER_AREA)
		
        try:
            resized_image = self.simulation_params.vision_bridge.cv2_to_imgmsg(cv_resized_image, "bgr8")
        except Exception as e:
            raise e
        self.ros_publisher.resized_color.publish(resized_image)
        return cv_resized_image
    
    def get_laser_observation(self) -> numpy.ndarray:
        scan = copy.deepcopy(self.simulation_params.scan)
        scan[numpy.isnan(scan)] = 30.0
        return scan
    
    def get_odom_speed(self) -> list:
        ve = numpy.sqrt(self.simulation_params.odometry.linear_x_speed ** 2 + self.simulation_params.odometry.linear_y_speed ** 2)
        return [ve, self.simulation_params.odometry.rotation_speed]
    
    def convert_states_to_state(self, states, object_name) -> ModelState:
        target_state = ModelState()
        source_states = copy.deepcopy(states)
        i = source_states.name.index(object_name)
        target_state.model_name = object_name
        target_state.pose = source_states.pose[i]
        target_state.twist = source_states.twist[i]
        target_state.reference_frame = 'world'
        return target_state
    
    def initialize_object_pose(self, object_name = 'mobile_base', use_semi_random_position = True) -> None:
        initial_seed = convert_euler_to_quaternion(numpy.random.uniform(-numpy.pi, numpy.pi), 0, numpy.pi)
		
        print()
        logger.info("initial_seed: " + str(initial_seed))
		
        if object_name is 'mobile_base' :
            object_state = copy.deepcopy(self.default_robot_state)
            if use_semi_random_position:
                spawn_location_key = random.randint(0, 3)
            else:
                spawn_location_key = K_SPAWNPOINT_CENTER
            
            if spawn_location_key == K_SPAWNPOINT_CENTER:
                object_state.pose.position.x =  0.5
                object_state.pose.position.y =  0.1
            elif spawn_location_key == K_SPAWNPOINT_UP_LEFT:
                object_state.pose.position.x = -2.5
                object_state.pose.position.y = -4.5
            elif spawn_location_key == K_SPAWNPOINT_UP_RIGHT:
                object_state.pose.position.x = -3.0
                object_state.pose.position.y =  4.5
            elif spawn_location_key == K_SPAWNPOINT_DOWN_LEFT:
                object_state.pose.position.x =  4.0
                object_state.pose.position.y = -4.5

            object_state.pose.orientation.x = initial_seed[0]
            object_state.pose.orientation.y = initial_seed[1]
            object_state.pose.orientation.z = initial_seed[2]
            object_state.pose.orientation.w = initial_seed[3]
        else:
            object_state = self.convert_states_to_state(self.simulation_params.default_states, object_name)
        self.ros_publisher.update_state.publish(object_state)
    
    def reset_environment(self) -> None:
        self.initialize_object_pose()
        
        for i in range(0, len(self.simulation_params.object_label), 1): 
            # INFO: support for dynamic obstacles and target destinations 
            self.initialize_object_pose(self.simulation_params.object_label[i])
        
        self.simulation_params.robot_speed = K_ROBOT_SPEED
        self.simulation_params.step_target = K_STEP_TARGET
        self.simulation_params.start_time = time.time()
        rospy.sleep(0.5)
    
    def publish_controller_input(self, action) -> None:
        if action < 2:
            self.simulation_params.robot_speed[0] = self.simulation_params.action_table[action]
        else:
            self.simulation_params.robot_speed[1] = self.simulation_params.action_table[action]
        
        move = Twist()
        move.linear.x   = self.simulation_params.robot_speed[0]
        move.linear.y   = 0.0
        move.linear.z   = 0.0
        move.angular.x  = 0.0
        move.angular.y  = 0.0
        move.angular.z  = self.simulation_params.robot_speed[1]
        self.ros_publisher.cmd_velocity.publish(move)
    
    def publish_depth_prediction(self, nn_depth_prediction) -> None:
        image_data = numpy.array(nn_depth_prediction, dtype = numpy.float32)
        try:
            resized_image = self.simulation_params.vision_bridge.cv2_to_imgmsg(image_data, "passthrough")
        except Exception as e:
            raise e
        self.ros_publisher.resized_depth.publish(resized_image)
    
    def calculate_reward(self, step):
        terminal = False
        should_reset = False
        
        [ve, theta] = self.get_odom_speed()
        laser = self.get_laser_observation()
        reward = ve * numpy.cos(theta) * 0.2 - 0.01

        if self.simulation_params.bump or numpy.amin(laser) < 0.46 or numpy.amin(laser) == 30.0:
            reward = -10.
            terminal = True
            should_reset = True
        if step > 500:
            should_reset = True
        return (reward, terminal, should_reset)


    
    
    