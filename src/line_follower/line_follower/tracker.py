#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from line_interfaces.msg import Line
import tf_transformations as tft
from .logging_framework import get_logger

#############
# CONSTANTS #
#############
_RATE = 10 # (Hz) rate for rospy.rate
_MAX_SPEED = 1.5 # (m/s)
_MAX_CLIMB_RATE = 1.0 # m/s
_MAX_ROTATION_RATE = 5.0 # rad/s 
IMAGE_HEIGHT = 576  # Updated to match actual cropped image dimensions
IMAGE_WIDTH = 768   # Updated to match actual cropped image dimensions
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2]) # Center of the image frame. We will treat this as the center of mass of the drone
EXTEND = 300 # Number of pixels forward to extrapolate the line
KP_X = 0.8    # Increased for more responsive lateral control
KP_Y = 0.8    # Increased for more responsive forward/backward control
KP_Z_W = 2.0  # Reduced to prevent oscillation
DISPLAY = True

#########################
# COORDINATE TRANSFORMS #
#########################
class CoordTransforms():

    def __init__(self):
        """
        Variable Notation:
            - v__x: vector expressed in "x" frame
            - q_x_y: quaternion of "x" frame with relative to "y" frame
            - p_x_y__z: position of "x" frame relative to "y" frame expressed in "z" coordinates
            - v_x_y__z: velocity of "x" frame with relative to "y" frame expressed in "z" coordinates
            - R_x2y: rotation matrix that maps vector represented in frame "x" to representation in frame "y" (right-multiply column vec)
    
        Frame Subscripts:
            - m = marker frame (x-right, y-up, z-out when looking at marker)
            - dc = downward-facing camera (if expressed in the body frame)
            - fc = forward-facing camera
            - bu = body up frame (x-forward, y-left, z-up, similar to ENU)
            - bd = body down frame (x-forward, y-right, z-down, similar to NED)
            - lenu = local East-North-Up world frame ("local" implies that it may not be aligned with east and north, but z is up)
            - lned = local North-East-Down world frame ("local" implies that it may not be aligned with north and east, but z is down)
        Rotation matrix:
            R = np.array([[       3x3     0.0]
                          [    rotation   0.0]
                          [     matrix    0.0]
                          [0.0, 0.0, 0.0, 0.0]])
            
            [[ x']      [[       3x3     0.0]  [[ x ]
             [ y']  =    [    rotation   0.0]   [ y ]
             [ z']       [     matrix    0.0]   [ z ]
             [0.0]]      [0.0, 0.0, 0.0, 0.0]]  [0.0]]
        """
        
        # Reference frames
        self.COORDINATE_FRAMES = {'lenu','lned','bu','bd','dc','fc'}
    
        self.WORLD_FRAMES = {'lenu', 'lned'}
    
        self.BODY_FRAMES = {'bu', 'bd', 'dc', 'fc'}
    
        self.STATIC_TRANSFORMS = {'R_lenu2lenu',
                                  'R_lenu2lned',
    
                                  'R_lned2lenu',
                                  'R_lned2lned', 
          
                                  'R_bu2bu', 
                                  'R_bu2bd',
                                  'R_bu2dc',
                                  'R_bu2fc',
          
                                  'R_bd2bu',
                                  'R_bd2bd',
                                  'R_bd2dc',
                                  'R_bd2fc',
          
                                  'R_dc2bu',
                                  'R_dc2bd',
                                  'R_dc2dc',
                                  'R_dc2fc',
    
                                  'R_fc2bu',
                                  'R_fc2bd',
                                  'R_fc2dc',
                                  'R_fc2fc'
                                  }
       
        # Define the transformation matrix from downward camera to body down frame
        self.R_dc2bd = np.array([
            [0.0, -1.0, 0.0, 0.0], # bd.x = -dc.y
            [1.0, 0.0, 0.0, 0.0],  # bd.y = dc.x
            [0.0, 0.0, 1.0, 0.0],  # bd.z = dc.z
            [0.0, 0.0, 0.0, 1.0]   # Fix: Last element should be 1.0, not 0.0
        ])
        
        # Define the identity matrices for same-frame transformations
        self.R_lenu2lenu = np.eye(4)
        self.R_lned2lned = np.eye(4) 
        self.R_bu2bu = np.eye(4)
        self.R_bd2bd = np.eye(4)
        self.R_dc2dc = np.eye(4)
        self.R_fc2fc = np.eye(4)
        
        # Define the transformation matrix from body down to downward camera frame
        self.R_bd2dc = np.array([
            [0.0, 1.0, 0.0, 0.0],  # dc.x = bd.y
            [-1.0, 0.0, 0.0, 0.0], # dc.y = -bd.x
            [0.0, 0.0, 1.0, 0.0],  # dc.z = bd.z
            [0.0, 0.0, 0.0, 1.0]   # Fix: Last element should be 1.0, not 0.0
        ])
    
    
    def static_transform(self, v__fin, fin, fout):
        """
        Given a vector expressed in frame fin, returns the same vector expressed in fout.
            
            Args:
                - v__fin: 3D vector, (x, y, z), represented in fin coordinates 
                - fin: string describing input coordinate frame 
                - fout: string describing output coordinate frame 
        
            Returns
                - v__fout: a vector, (x, y, z) represent in fout coordinates
        """
        # Check if fin is a valid coordinate frame
        if fin not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fin))

        # Check if fout is a valid coordinate frame
        if fout not in self.COORDINATE_FRAMES:
            raise AttributeError('{} is not a valid coordinate frame'.format(fout))
        
        # Check for a static transformation exists between the two frames
        R_str = 'R_{}2{}'.format(fin, fout)
        if R_str not in self.STATIC_TRANSFORMS:
            raise AttributeError('No static transform exists from {} to {}.'.format(fin, fout))
        
        # v4__'' are 4x1 np.array representation of the vector v__''
        # Create a 4x1 np.array representation of v__fin for matrix multiplication
        v4__fin = np.array([[v__fin[0]],
                            [v__fin[1]],
                            [v__fin[2]],
                            [     0.0]])

        # Get rotation matrix
        R_fin2fout = getattr(self, R_str)

        # Perform transformation from v__fin to v__fout
        v4__fout = np.dot(R_fin2fout, v4__fin)
        
        return (v4__fout[0,0], v4__fout[1,0], v4__fout[2,0])


class LineController(Node):
    def __init__(self) -> None:
        super().__init__('line_controller')

        # Initialize custom logger
        self.logger = get_logger('tracker')

        # Create CoordTransforms instance
        self.coord_transforms = CoordTransforms()

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.line_sub = self.create_subscription(
            Line, '/line/param', self.line_sub_cb, 1)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -3.0

        # Linear setpoint velocities in downward camera frame
        self.vx__dc = 0.0
        self.vy__dc = 0.0
        self.vz__dc = 0.0

        # Yaw setpoint velocities in downward camera frame
        self.wz__dc = 0.0

        # Quaternion representing the rotation of the drone's body frame in the NED frame. initiallize to identity quaternion
        self.quat_bu_lenu = (0, 0, 0, 1)

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.logger.system('system_events', 'Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.logger.system('system_events', 'Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.logger.system('system_events', "Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.logger.system('system_events', "Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self, vx: float, vy: float, wz: float) -> None:
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [None, None, self.takeoff_height]
        if self.offboard_setpoint_counter < 100:
            msg.velocity = [0.0, 0.0, 0.0]
        else:
            msg.velocity = [vx, vy, 0.0]
        msg.acceleration = [None, None, None]
        msg.yawspeed = wz
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.logger.debug('velocity_commands', f"Publishing velocity setpoints {[vx, vy, wz]}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def convert_velocity_setpoints(self):
        # Debug input values
        self.logger.debug('coordinate_transform', f"Converting DC velocities: vx={self.vx__dc:.3f}, vy={self.vy__dc:.3f}, wz={self.wz__dc:.3f}")
        
        # Set linear velocity (convert command velocity from downward camera frame to bd frame)
        vx, vy, vz = self.coord_transforms.static_transform((self.vx__dc, self.vy__dc, self.vz__dc), 'dc', 'bd')

        # Set angular velocity (convert command angular velocity from downward camera to bd frame)
        _, _, wz = self.coord_transforms.static_transform((0.0, 0.0, self.wz__dc), 'dc', 'bd')

        self.logger.debug('coordinate_transform', f"After coordinate transform: vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")

        # enforce safe velocity limits
        if _MAX_SPEED < 0.0 or _MAX_CLIMB_RATE < 0.0 or _MAX_ROTATION_RATE < 0.0:
            raise Exception("_MAX_SPEED,_MAX_CLIMB_RATE, and _MAX_ROTATION_RATE must be positive")
        vx = min(max(vx,-_MAX_SPEED), _MAX_SPEED)
        vy = min(max(vy,-_MAX_SPEED), _MAX_SPEED)
        wz = min(max(wz,-_MAX_ROTATION_RATE), _MAX_ROTATION_RATE)

        self.logger.debug('velocity_commands', f"After velocity limits: vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")
        return (vx, vy, wz)
    
    def timer_callback(self) -> None:
        """Callback function for the timer."""

        self.publish_offboard_control_heartbeat_signal()
        
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        self.offboard_setpoint_counter += 1
    
    def line_sub_cb(self, param):
        """
        Callback function which is called when a new message of type Line is recieved by self.line_sub.
        Notes:
        - This is the function that maps a detected line into a velocity 
        command
            
            Args:
                - param: parameters that define the center and direction of detected line
        """
        self.logger.info("line_detection", "Line detected, following line")
        # Extract line parameters
        x, y, vx, vy = param.x, param.y, param.vx, param.vy
        
        # Normalize direction vector to ensure it's a unit vector
        norm = np.sqrt(vx**2 + vy**2)
        if norm > 0:
            vx /= norm
            vy /= norm
            
        # Target point is EXTEND pixels ahead along the line
        target_x = x + EXTEND * vx
        target_y = y + EXTEND * vy
        
        # Calculate error between target and center of image
        error_x = target_x - CENTER[0]
        error_y = target_y - CENTER[1]
        
        # Debug info - show what we received
        self.logger.debug("line_detection", f"Line at ({x:.1f}, {y:.1f}), direction ({vx:.2f}, {vy:.2f})")
        self.logger.debug("line_detection", f"Target at ({target_x:.1f}, {target_y:.1f}), errors ({error_x:.1f}, {error_y:.1f})")
        
        # SIMPLIFIED CONTROL LOGIC - direct proportional control
        # For lateral control: error_x directly controls lateral movement
        self.vx__dc = KP_X * error_x / 100.0  # Right/left movement based on target position
        
        # For forward movement: always move forward, adjust based on y error
        base_forward_speed = 0.5  # Constant forward movement
        y_correction = -KP_Y * error_y / 200.0  # Adjust forward speed based on line position
        self.vy__dc = -(base_forward_speed + y_correction)  # Negative for forward in dc frame
        
        # For heading control: align with line direction
        desired_heading = np.arctan2(vx, -vy)  # Calculate desired heading from line direction
        angular_error = self.normalize_angle(desired_heading)
        self.wz__dc = KP_Z_W * angular_error / 10.0  # Reduce oscillation
        
        # Debug control outputs before coordinate transformation
        self.logger.debug("control_loops", f"DC frame commands: vx={self.vx__dc:.3f}, vy={self.vy__dc:.3f}, wz={self.wz__dc:.3f}")
        
        # Convert and publish velocity commands
        vx_bd, vy_bd, wz_bd = self.convert_velocity_setpoints()
        self.logger.debug("control_loops", f"BD frame commands: vx={vx_bd:.3f}, vy={vy_bd:.3f}, wz={wz_bd:.3f}")
        self.publish_trajectory_setpoint(vx_bd, vy_bd, wz_bd)
    
    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

def main(args=None) -> None:
    logger = get_logger('system')
    logger.system('startup', 'Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = LineController()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger = get_logger('system')
        logger.error('errors', str(e))