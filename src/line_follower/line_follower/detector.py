#!/usr/bin/env python
# adapted from fitline algorithm by sonia

###########
# IMPORTS #
###########
from matplotlib import image
import numpy as np
import rclpy
from rclpy.node import Node
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from line_interfaces.msg import Line
import sys
from .linreg import process_linreg
from line_follower.logger import logging_framework
from line_follower.line_visualizer import create_line_following_debug_image

#############
# CONSTANTS #
#############
LOW = None  # Lower image thresholding bound
HI = None   # Upper image thresholding bound
LENGTH_THRESH = None  # If the length of the largest contour is less than LENGTH_THRESH, we will not consider it a line
KERNEL = np.ones((5, 5), np.uint8)
DISPLAY = True
# Define FOV reduction factor (0.5 = 50% reduction)
FOV_REDUCTION = 0.6

logger = logging_framework.Logger()
class LineDetector(Node):
    def __init__(self):
        super().__init__('detector')

        # A subscriber to the topic '/aero_downward_camera/image'
        self.camera_sub = self.create_subscription(
            Image,
            '/world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image',
            self.camera_sub_cb,
            10
        )

        self.cvresult_pub = self.create_publisher(Image, '/line/cvresult', 10)

        # A publisher which will publish a parametrization of the detected line to the topic '/line/param'
        self.param_pub = self.create_publisher(Line, '/line/param', 1)

        # A publisher which will publish an image annotated with the detected line to the topic 'line/detector_image'
        self.detector_image_pub = self.create_publisher(Image, '/line/detector_image', 1)

        # Initialize instance of CvBridge to convert images between OpenCV images and ROS images
        self.bridge = CvBridge()

        # Add a counter for consecutive detection failures
        self.no_detection_count = 0
        # Maximum allowed failures before using default line
        self.max_no_detection = 10
        
        # Store original image dimensions and center for reference
        self.original_width = None
        self.original_height = None
        self.original_center = None
        
        # Direction consistency tracking
        self.previous_direction = None  # Store previous direction vector (vx, vy)
        self.detection_count = 0       # Count of successful detections for startup handling

    def crop_center(self, img, crop_factor=FOV_REDUCTION):
        """
        Crop the center portion of the image to simulate a reduced FOV.
        Args:
            img: Input image to crop
            crop_factor: Factor to crop (0.5 means keep 50% of original size)
        Returns:
            Cropped image
        """
        height, width = img.shape[:2]
        
        # Store original dimensions if not already stored
        if self.original_width is None or self.original_height is None:
            self.original_width = width
            self.original_height = height
            self.original_center = (width // 2, height // 2)
        
        # Calculate new dimensions
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)
        
        # Calculate crop coordinates
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2
        end_x = start_x + new_width
        end_y = start_y + new_height
        
        # Crop the image
        cropped_img = img[start_y:end_y, start_x:end_x]
        
        return cropped_img

    def calculate_angular_distance(self, v1, v2):
        """
        Calculate the angular distance between two 2D vectors.
        Args:
            v1, v2: Tuples representing 2D vectors (vx, vy)
        Returns:
            Angular distance in radians (0 to pi)
        """
        # Calculate dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # Calculate magnitudes
        mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return np.pi  # Maximum distance if one vector is zero
        
        # Calculate cosine of angle
        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        
        # Return angle in radians (0 to pi)
        # Remove abs() to properly distinguish between same and opposite directions
        return np.arccos(cos_angle)

    def choose_consistent_direction(self, vx, vy):
        # enforce forward (-y in camera frame) consistency
        b4_vx, b4_vy = vx, vy
        if vy > 0:
            # If vy is positive, the vector is pointing "down" in the image.
            return (-vx, -vy)
        
        # If vy is positive or zero, the direction is already correct.
        logger.log("detector_flip_debug", logging_framework.LoggerColors.YELLOW, f"Choosing consistent direction. Current: ({b4_vx}, {b4_vy}), New: ({vx}, {vy})", 1000)
        return (vx, vy)

    ######################
    # CALLBACK FUNCTIONS #
    ######################
    def camera_sub_cb(self, msg):
        """
        Callback function which is called when a new message of type Image is received by self.camera_sub.
            Args: 
                - msg = ROS Image message
        """
        # Convert Image msg to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        
        # Crop the image to reduce FOV
        cropped_image = self.crop_center(image)
        
        # Detect line in the cropped image
        line = self.detect_line(cropped_image, msg)

        # If no line was detected, increment counter and use default if needed
        if line is None:
            self.no_detection_count += 1
            if self.no_detection_count > self.max_no_detection:
                # Use default line (center of cropped image) with consistent direction
                logger.log("detector", logging_framework.LoggerColors.YELLOW, "Using default line after multiple detection failures", 1000)
                default_vx, default_vy = 0.0, 1.0  # Default downward direction
                
                # Apply direction consistency if we have previous direction
                if self.previous_direction is not None and self.detection_count >= 3:
                    default_vx, default_vy = self.choose_consistent_direction(default_vx, default_vy)
                    self.previous_direction = (default_vx, default_vy)
                
                line = (cropped_image.shape[1]/2, cropped_image.shape[0]/2, default_vx, default_vy)
        else:
            # Reset counter when line is detected
            self.no_detection_count = 0

        # If a line was detected or default is used, publish the parameterization
        if line is not None:
            msg = Line()
            msg.x = float(line[0])
            msg.y = float(line[1])
            msg.vx = float(line[2])
            msg.vy = float(line[3])
            # Publish param msg
            self.param_pub.publish(msg)

        # Publish annotated image if DISPLAY is True and a line was detected
        if DISPLAY and line is not None:
            # Draw the detected line on a color version of the image
            annotated = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
            x, y, vx, vy = line
            pt1 = (int(x - 100*vx), int(y - 100*vy))
            pt2 = (int(x + 100*vx), int(y + 100*vy))
            cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
            # Convert to ROS Image message and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            self.detector_image_pub.publish(annotated_msg)

    ##########
    # DETECT #
    ##########
    def detect_line(self, image, msg):
        """ 
        Given an image, fit a line to biggest contour if it meets size requirements (otherwise return None)
        and return a parameterization of the line as a center point on the line and a vector
        pointing in the direction of the line.
            Args:
                - image = OpenCV image
            Returns: (x, y, vx, vy) where (x, y) is the centerpoint of the line in image and 
            (vx, vy) is a vector pointing in the direction of the line. Both values are given
            in downward camera pixel coordinates. Returns None if no line is found
        """

        if len(image.shape) == 2 or image.shape[2] == 1:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        # Process with BGR image
        img, line_info = process_linreg(image_bgr)

        
        if line_info is None:
            logger.log("detector", logging_framework.LoggerColors.RED, "No line information received from line processor", 1000)
            return None
            
        # Comprehensive check for invalid line data
        if ('x_position' not in line_info or 
            'slope' not in line_info or 
            'intercept' not in line_info or 
            'is_vertical' not in line_info):
            logger.log("detector_warn", logging_framework.LoggerColors.YELLOW, "Invalid line information received from line processor", 1000)
            return None
            
        # Extract values from line_info - now with enhanced curve support
        x0 = line_info['x_position']
        y0 = line_info.get('y_position', image.shape[0] / 2)  # Use provided y or default to center
        m = line_info['slope']
        b = line_info['intercept']
        vertical = line_info['is_vertical']
        
        # Check if we have direct direction vectors from curve fitting
        if 'direction_x' in line_info and 'direction_y' in line_info:
            # Use the direction vectors directly from the curve fitting
            vx = line_info['direction_x']
            vy = line_info['direction_y']
            x = x0
            y = y0

        else:
            # Fallback to traditional slope-based calculation
            if vertical:
                # For a vertical line, direction is straight down (0, 1)
                x = x0
                y = image.shape[0] / 2
                vx = 0.0
                vy = 1.0  # Point down in the image
            else:
                # For a non-vertical line, calculate direction using slope
                x = image.shape[1] / 2  # Center x position
                y = m * x + b          # Corresponding y position
                
                # Get a point further along the line (in positive x direction)
                x2 = x + 100
                y2 = m * x2 + b
                
                # Direction vector from (x,y) to (x2,y2)
                vx = x2 - x
                vy = y2 - y
        
        # Check if any of the line parameters are NaN
        if (np.isnan(x) or np.isnan(y) or np.isnan(vx) or np.isnan(vy)):
            logger.log("detector_warn", logging_framework.LoggerColors.YELLOW, "Detected line parameters contain NaN values", 1000)
            return None
        
        # Normalize direction vector
        norm = np.sqrt(vx**2 + vy**2)
        if norm > 0:
            vx /= norm
            vy /= norm
        
        # Choose direction that maintains consistency with previous movement
        vx, vy = self.choose_consistent_direction(vx, vy)
        resulting_image = create_line_following_debug_image(
        image=img,
        line_x=x0, line_y=y0, line_vx=vx, line_vy=vy,
        vx_dc=vx, vy_dc=vy, wz_dc=0.0,
        # Additional debug info
        error_x=line_info['error_x'],
        error_y=line_info['error_y'],
        confidence=line_info['confidence'],
        method=line_info['method'],
        points_detected=line_info['points_detected']
    )
        linreg_msg = self.bridge.cv2_to_imgmsg(resulting_image, encoding='bgr8')
        linreg_msg.header = msg.header
        self.cvresult_pub.publish(linreg_msg)
        # Store this direction for next iteration and increment detection count
        self.previous_direction = (vx, vy)
        self.detection_count += 1

        logger.log("detector", logging_framework.LoggerColors.GREEN, f"Detected line: x={x}, y={y}, vx={vx}, vy={vy} (detection #{self.detection_count})", 1000)
        return (x, y, vx, vy)


            
def main(args=None):
    rclpy.init(args=args)
    detector = LineDetector()
    detector.get_logger().info("Line detector initialized")
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception as e:
        print(e)
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()