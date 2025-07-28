"""
Robot integration module for OpenVLA
Provides interfaces for various robot platforms and simulation environments
"""

import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Tuple, Any
import threading
from abc import ABC, abstractmethod
import json
import os

# Optional imports for different robot platforms
try:
    import rospy
    from sensor_msgs.msg import Image as ROSImage
    from geometry_msgs.msg import Pose, Twist
    from std_msgs.msg import Float64MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

class RobotInterface(ABC):
    """
    Abstract base class for robot interfaces
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the robot"""
        pass
    
    @abstractmethod
    def get_camera_image(self) -> np.ndarray:
        """Get current camera image"""
        pass
    
    @abstractmethod
    def execute_action(self, action: np.ndarray) -> bool:
        """Execute robot action"""
        pass
    
    @abstractmethod
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        pass

class PyBulletRobot(RobotInterface):
    """
    PyBullet simulation robot interface
    """
    
    def __init__(self, robot_urdf: str = "panda.urdf", gui: bool = True):
        self.robot_urdf = robot_urdf
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.camera_distance = 1.5
        self.camera_yaw = 50
        self.camera_pitch = -35
        self.camera_target_position = [0.7, 0, 0.5]
        
    def connect(self) -> bool:
        """Connect to PyBullet simulation"""
        if not PYBULLET_AVAILABLE:
            raise ImportError("PyBullet not available. Install with: pip install pybullet")
        
        # Connect to PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load robot
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf, start_pos, start_orientation)
        
        # Set up camera
        self._setup_camera()
        
        return True
    
    def _setup_camera(self):
        """Set up camera view"""
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target_position
        )
    
    def disconnect(self):
        """Disconnect from PyBullet"""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            self.robot_id = None
    
    def get_camera_image(self) -> np.ndarray:
        """Get camera image from PyBullet"""
        if self.robot_id is None:
            raise RuntimeError("Robot not connected")
        
        # Get camera view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target_position,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        # Get projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Get camera image
        width, height = 640, 480
        image_arr = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Extract RGB image
        rgb = image_arr[2]  # RGB data
        rgb = np.reshape(rgb, (height, width, 4))  # RGBA
        rgb = rgb[:, :, :3]  # Remove alpha channel
        
        return rgb
    
    def execute_action(self, action: np.ndarray) -> bool:
        """Execute action in PyBullet simulation"""
        if self.robot_id is None:
            raise RuntimeError("Robot not connected")
        
        # Parse action (assuming 7-DoF: 3 position + 4 quaternion)
        if len(action) == 7:
            # Position control
            target_pos = action[:3]
            target_quat = action[3:7]
            
            # Set joint positions (simplified - in practice you'd use IK)
            joint_positions = self._inverse_kinematics(target_pos, target_quat)
            
            # Set joint positions
            for i, pos in enumerate(joint_positions):
                p.resetJointState(self.robot_id, i, pos)
            
            # Step simulation
            p.stepSimulation()
            time.sleep(0.01)  # Small delay for visualization
            
        return True
    
    def _inverse_kinematics(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """Simple inverse kinematics (placeholder)"""
        # This is a simplified IK - in practice you'd use proper IK solver
        num_joints = p.getNumJoints(self.robot_id)
        joint_positions = np.zeros(num_joints)
        
        # Simple heuristic IK
        for i in range(num_joints):
            joint_positions[i] = np.random.uniform(-np.pi, np.pi)
        
        return joint_positions
    
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        if self.robot_id is None:
            raise RuntimeError("Robot not connected")
        
        # Get joint states
        joint_states = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointState(self.robot_id, i)
            joint_states.append({
                'position': joint_info[0],
                'velocity': joint_info[1],
                'reaction_force': joint_info[2],
                'applied_torque': joint_info[3]
            })
        
        # Get end effector pose
        end_effector_link = p.getNumJoints(self.robot_id) - 1
        end_effector_state = p.getLinkState(self.robot_id, end_effector_link)
        
        return {
            'joint_states': joint_states,
            'end_effector_position': end_effector_state[0],
            'end_effector_orientation': end_effector_state[1],
            'end_effector_linear_velocity': end_effector_state[2],
            'end_effector_angular_velocity': end_effector_state[3]
        }
    
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        return self.robot_id is not None

def create_robot_interface(robot_type: str, **kwargs) -> RobotInterface:
    """
    Factory function to create robot interface
    
    Args:
        robot_type: Type of robot ('pybullet', 'ros', 'real')
        **kwargs: Additional arguments for robot interface
    
    Returns:
        RobotInterface instance
    """
    if robot_type == 'pybullet':
        return PyBulletRobot(**kwargs)
    elif robot_type == 'ros':
        # Placeholder for ROS interface
        raise NotImplementedError("ROS interface not implemented yet")
    elif robot_type == 'real':
        # Placeholder for real robot interface
        raise NotImplementedError("Real robot interface not implemented yet")
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

# Example usage
if __name__ == "__main__":
    # Create PyBullet robot interface
    robot = create_robot_interface('pybullet', gui=True)
    
    # Connect to robot
    if robot.connect():
        print("Connected to PyBullet robot")
        
        # Get camera image
        image = robot.get_camera_image()
        print(f"Camera image shape: {image.shape}")
        
        # Execute test action
        test_action = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])  # 7-DoF action
        robot.execute_action(test_action)
        
        # Get robot state
        state = robot.get_robot_state()
        print(f"Robot state: {state}")
        
        # Disconnect
        robot.disconnect()
    else:
        print("Failed to connect to robot") 