#!/usr/bin/env python

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
from .logging_framework import Logger, LoggerColors

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

class LineDetector(Node):
    def __init__(self):
        super().__init__('detector')

        # Initialize custom logger
        self.logger = Logger()

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
        """
        Choose the direction vector that maintains consistency with previous movement.
        A line can point in two directions, so we pick the one closest to previous direction.
        Args:
            vx, vy: Current direction vector components
        Returns:
            (vx, vy) tuple with consistent direction
        """
        # For the first few detections or if no previous direction, use downward preference
        if self.previous_direction is None or self.detection_count < 3:
            # Default behavior: prefer downward direction in image
            if vy < 0:
                return (-vx, -vy)
            return (vx, vy)
        
        # Calculate angular distances for both possible directions
        direction1 = (vx, vy)
        direction2 = (-vx, -vy)
        
        dist1 = self.calculate_angular_distance(direction1, self.previous_direction)
        dist2 = self.calculate_angular_distance(direction2, self.previous_direction)
        
        # Choose the direction with smaller angular distance
        if dist1 <= dist2:
            return direction1
        else:
            return direction2

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

        # Process with BGR image (no heading available in detector, use default)
        resulting_image, line_info = process_linreg(image_bgr, 0.0)
        linreg_msg = self.bridge.cv2_to_imgmsg(resulting_image, encoding='bgr8')
        linreg_msg.header = msg.header
        self.cvresult_pub.publish(linreg_msg)
        if line_info is None:
            return None
            
        # Comprehensive check for invalid line data
        if ('x_position' not in line_info or 
            'slope' not in line_info or 
            'intercept' not in line_info or 
            'is_vertical' not in line_info):
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
            return None
        
        # Normalize direction vector
        norm = np.sqrt(vx**2 + vy**2)
        if norm > 0:
            vx /= norm
            vy /= norm
        
        # Choose direction that maintains consistency with previous movement
        vx, vy = self.choose_consistent_direction(vx, vy)
        
        # Store this direction for next iteration and increment detection count
        self.previous_direction = (vx, vy)
        self.detection_count += 1
        
        return (x, y, vx, vy)


            
def main(args=None):
    rclpy.init(args=args)
    detector = LineDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()