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

#############
# CONSTANTS #
#############
LOW = None  # Lower image thresholding bound
HI = None   # Upper image thresholding bound
LENGTH_THRESH = None  # If the length of the largest contour is less than LENGTH_THRESH, we will not consider it a line
KERNEL = np.ones((5, 5), np.uint8)
DISPLAY = True
# Define FOV reduction factor (0.5 = 50% reduction)
FOV_REDUCTION = 0.5

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
            self.get_logger().info(f"Original image dimensions: {width}x{height}")
        
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
        self.get_logger().info(f"Cropped image to {new_width}x{new_height} (FOV reduced by {(1-crop_factor)*100}%)")
        
        return cropped_img

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
                # Use default line (center of cropped image, pointing down)
                self.get_logger().warn("Using default line after multiple detection failures")
                line = (cropped_image.shape[1]/2, cropped_image.shape[0]/2, 0.0, 1.0)
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
            self.get_logger().info(f"Published line: x={line[0]}, y={line[1]}, vx={line[2]}, vy={line[3]}")

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
        resulting_image, line_info = process_linreg(image_bgr)
        linreg_msg = self.bridge.cv2_to_imgmsg(resulting_image, encoding='bgr8')
        linreg_msg.header = msg.header
        self.cvresult_pub.publish(linreg_msg)
        if line_info is None:
            self.get_logger().warn("No line detected in image")
            return None
            
        # Comprehensive check for invalid line data
        if ('x_position' not in line_info or 
            'slope' not in line_info or 
            'intercept' not in line_info or 
            'is_vertical' not in line_info):
            self.get_logger().warn("Invalid line information received from line processor")
            return None
            
        # Extract values from line_info
        x0 = line_info['x_position']
        m = line_info['slope']
        b = line_info['intercept']
        vertical = line_info['is_vertical']
        
        # Calculate the direction vector (vx, vy) based on line properties
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
            self.get_logger().warn("Invalid line parameters detected (NaN values)")
            return None
        
        # Normalize direction vector
        norm = np.sqrt(vx**2 + vy**2)
        if norm > 0:
            vx /= norm
            vy /= norm
        
        # Make sure the vector points downward in the image (positive y)
        # This ensures consistent line following direction
        if vy < 0:
            vx = -vx
            vy = -vy
        
        self.get_logger().info(f"Detected line: x={x}, y={y}, vx={vx}, vy={vy}")
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