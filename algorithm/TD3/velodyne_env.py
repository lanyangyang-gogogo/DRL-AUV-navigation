import math
import os
import random
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# Maximum distance between the goal and the robot
GOAL_REACHED_DIST = 4.5    
# Safe distance between the robot and obstacles
COLLISION_DIST = 1.5    
TIME_DELTA = 0.1

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y, z):
    goal_ok = True
    if x > 25 or x < -25 or y > 25 or y < -25 or z != -50:
        goal_ok = False
    return goal_ok

def check_pos_st(x, y, z):
    goal_ok = True
    # Check if x and y are within the valid starting range
    if not (-20 <= x <= 20 and 0 <= y <= 20 and z == -50):
        goal_ok = False

    if (2 < x < 18 and 2 < y < 18):
        goal_ok = False

    if (-18 < x < -2 and 2 < y < 18):
        goal_ok = False

    return goal_ok

def check_pos_go(x, y, z):
    goal_ok = True
    if not (-20 <= x <= 20 and -20 <= y <= 0 and z == -50):
        goal_ok = False

    if (2 < x < 18 and -18 < y < -2):
        goal_ok = False

    if (-18 < x < -2 and -18 < y < -2):
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""
    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        # Initial coordinates
        self.odom_x = 0
        self.odom_y = 0
        self.odom_z = -50

        self.goal_x = 5
        self.goal_y = 0.0

        # Store the initial coordinates at the start of the episode
        self.initial_robot_x = 0.0
        self.initial_robot_y = 0.0
        self.initial_goal_x = 0.0
        self.initial_goal_y = 0.0

        # Upper and lower bounds
        self.upper = 20
        self.lower = -20
        self.velodyne_data = np.ones(self.environment_dim) * 50
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "rexrov"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = -50.0
        
        # Initial quaternion orientation of the robot
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        # Quaternion representing the spatial orientation
        self.set_self_state.pose.orientation.w = 1.0
        
        # -0.03 is a small offset used to avoid edge cases and ensure necessary margin
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        # ==========================================================
        # Decoupled Architecture: Wait for external ROS environment
        # ==========================================================
        
        # Initialize the ROS node (it will connect to the roscore launched in the other terminal)
        rospy.init_node("gym", anonymous=True)
        print("ROS node initialized. Waiting for external Gazebo simulation to be ready...")

        # Block and wait for core ROS services to ensure Gazebo is fully launched
        rospy.wait_for_service("/gazebo/unpause_physics")
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/reset_world")
        print("Gazebo services connected successfully!")

        # Set up the ROS publishers and subscribers
        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher("/rexrov/cmd_vel", Twist, queue_size=1)
        # Gazebo model state publisher
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        # 3 DoF
        self.publisher4 = rospy.Publisher("pitch_velocity", MarkerArray, queue_size=1)
        
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/rexrov/pose_gt", Odometry, self.odom_callback, queue_size=1
        )

        # Block and wait for the first frame of sensor data to prevent state retrieval errors
        print("Waiting for sensor data (Odometry and PointCloud2)...")
        rospy.wait_for_message("/rexrov/pose_gt", Odometry)
        rospy.wait_for_message("/velodyne_points", PointCloud2)
        print("All sensor data received. Environment initialization complete. Ready to start training!")
        # ==========================================================

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    # Read data: Extract (x, y, z) coordinates from Velodyne point cloud.
    # Filter data: Only process points where -2 < z < 2.
    # Calculate angle: Calculate the angle beta relative to the reference vector (1, 0).
    # Calculate distance: Calculate the distance dist from each point to the origin.
    # Update distance: Update the minimum distance for each angle range in velodyne_data.
    def velodyne_callback(self, v):
        # Extract point cloud data containing (x, y, z) coordinates
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
         
        # velodyne_data: Stores the minimum distance for each angle range
        self.velodyne_data = np.ones(self.environment_dim) * 50

        for i in range(len(data)):
                if -2 < data[i][2] < 2:
                    dot = data[i][0] * 1 + data[i][1] * 0
                    mag1 = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2)
                    mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
         
                    # Check if mag1 is zero to avoid division by zero
                    if mag1 == 0:
                        print(f"Warning: mag1 is zero at point {data[i]}. Setting default values.")
                        continue  # Skip the current point
         
                    beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                    dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
         
                    for j in range(len(self.gaps)):
                        if self.gaps[j][0] <= beta < self.gaps[j][1]:
                            self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                            break


    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]   # Linear velocity
        vel_cmd.angular.z = action[1]  # Angular velocity (yaw)
        vel_cmd.angular.y = action[2]  # Angular velocity (pitch) - 3 DoF
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # Propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # Read velodyne laser state
        # Observe collision status from lidar data
        done, collision, min_laser = self.observe_collision(self.velodyne_data)

        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        # Related to degrees of freedom (3 DoF in the plane)
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        self.odom_z = self.last_odom.pose.pose.position.z
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)   # Convert quaternion to Euler angles
        angle = round(euler[2], 4)   # Robot's yaw angle
        pitch = round(euler[1], 4)

        # Calculate the Euclidean distance from the robot to the goal
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        # Calculate the coordinate offsets (skew) between the goal and the robot on the x and y axes
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0    # Dot product with the reference heading vector [1, 0]
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))  # Magnitude of the skew vector
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))   # Calculate the angle beta using the arccosine function

        if skew_y < 0:   # If skew_y < 0 (goal is below the robot), adjust the angle
            if skew_x < 0:
                beta = -beta  # Goal is to the bottom-left
            else:
                beta = 0 - beta  # Goal is to the bottom-right
        theta = beta - angle   # Angle difference between current heading and goal heading
        
        # Normalize the angle to [-pi, pi]
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
            print("Goal reached! Episode done.")

        robot_state = [distance, theta, action[0], action[1], action[2]]   # 3 DoF

        # State includes lidar data and the robot's state; Action includes linear and angular velocities
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(self, target, collision, action, min_laser, theta, distance)
        return state, reward, done, target

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # Generate a random yaw angle in [-pi, pi] for the initial orientation
        angle = np.random.uniform(-np.pi, np.pi)
        pitch = 0.0
        
        # Convert Euler angles to quaternion (roll and pitch are 0)
        quaternion = Quaternion.from_euler(0.0, pitch, angle) # 3 DoF
        object_state = self.set_self_state

        x = 0
        y = 0
        z = -50
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-25, 25)  # Randomly generate x coordinate between -25 and 25
            y = np.random.uniform(-25, 25)  # Randomly generate y coordinate
            position_ok = check_pos_st(x, y, z)     # Check if the coordinates are valid
            
        object_state.pose.position.x = x  # Set valid coordinates
        object_state.pose.position.y = y
        object_state.pose.position.z = z
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x   # Update current odometry
        self.odom_y = object_state.pose.position.y

        # Set a random goal in empty space in environment
        self.change_goal()

        # Store the initial coordinates at the start of the episode
        self.initial_robot_x = self.odom_x
        self.initial_robot_y = self.odom_y
        self.initial_goal_x = self.goal_x
        self.initial_goal_y = self.goal_y
        
        # Calculate initial Euclidean distance
        initial_distance = np.linalg.norm([
            self.initial_robot_x - self.initial_goal_x,
            self.initial_robot_y - self.initial_goal_y
        ])

        # Randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0, 0.0])  # 3 DoF

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Straight-line distance from current robot position to the goal
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        # Coordinate offsets between the goal and the robot
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0, 0.0] # 3 DoF
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        # Dynamically expand the boundaries to allow generating goals further away over time
        if self.upper < 50:    
            self.upper += 0.004
        if self.lower > -50:
            self.lower -= 0.004

        goal_ok = False

        # Loop until a valid goal position is found
        while not goal_ok:
            # Boundaries control the offset range for the new goal
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            self.goal_z = -50
            goal_ok = check_pos_go(self.goal_x, self.goal_y, self.goal_z)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(3):    # Create 3 boxes
            name = "random_box_" + str(i)    # Base name for the boxes defined in TD3.world

            x = 0
            y = 0
            z = -50
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-25, 25)   # Box generation range
                y = np.random.uniform(-25, 25)
                box_ok = check_pos(x, y, z)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                # Invalid position if the box is too close to the robot or the goal (distance < 10)
                if distance_to_robot < 10 or distance_to_goal < 10:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = -50.0  # Box z-coordinate is set to -50.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0  # Set box orientation (no rotation)
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = self.goal_z

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "world"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "world"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

        markerArray4 = MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "world"
        marker4.type = marker.CUBE
        marker4.action = marker.ADD
        marker4.scale.x = abs(action[2])
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 1.0
        marker4.color.g = 0.0
        marker4.color.b = 0.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = 5
        marker4.pose.position.y = 0.2
        marker4.pose.position.z = 0.2

        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            print("Collision detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(self, target, collision, action, min_laser, theta, distance):
        SAFE_DIST = 3.0  # Safe distance threshold (meters); triggers penalty if closer
        CLOSE_PENALTY_WEIGHT = 10.0  # Weight for close-distance penalty
        time_step_penalty = -0.03  # Time step penalty
        
        # 1. Get core distance variables
        initial_distance = np.linalg.norm([
            self.initial_robot_x - self.initial_goal_x,
            self.initial_robot_y - self.initial_goal_y
        ])
        current_distance = distance  # Current real-time distance
        
        # 2. Calculate approach goal reward factor (normalized to 0-0.5)
        approach_reward = 0.0
        # Avoid division by zero if initial distance is 0
        if initial_distance > 1e-6:
            # Distance change rate: (initial - current) / initial
            distance_rate = (initial_distance - current_distance) / initial_distance
            # Bound limits: no reward for moving away (negative), max 1 (reached goal)
            distance_rate = max(-1.0, min(1.0, distance_rate))
            # Normalize to [0, 0.5]
            approach_reward = distance_rate * 0.5
            # Extra optimization: give max approach reward if within goal threshold
            if current_distance < 4.5:
                approach_reward = 0.5

        # 3. Heading reward (normalized to 0-0.5 weight)
        # theta range is [-pi, pi], abs(theta) range is [0, pi]
        # Normalize to [0, 1]: 1 is best heading, 0 is worst
        normalized_heading = 1 - 2 * (abs(theta) / np.pi)
        # Apply weight of 0.5
        heading_reward = normalized_heading * 0.5
        
        # 4. Original base reward calculation
        base_reward = 0.0
        if target:
            base_reward = 300.0  # Base reward for reaching the goal
        elif collision:
            base_reward = -250.0  # Penalty for collision
        else:
            r3 = lambda x: 10 - x if x < 10 else 0.0
            base_reward = action[0] - abs(action[1]) / 2 - r3(min_laser) / 20  # Original continuous base reward

            # Add non-linear close-distance penalty
            if min_laser < SAFE_DIST:
                # Penalty formula ensures steep penalty for getting too close
                close_penalty = (SAFE_DIST - min_laser) **2 * CLOSE_PENALTY_WEIGHT
                base_reward -= close_penalty  # Subtract penalty from base reward
            
            # Add time step penalty
            base_reward += time_step_penalty
        
        # 5. Total reward = base + approach + heading
        total_reward = base_reward + approach_reward + heading_reward
        return total_reward
