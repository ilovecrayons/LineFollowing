import rclpy
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from px4_msgs.msg import VehicleLocalPosition
import cv2
from cv_bridge import CvBridge 
import numpy as np
from line_follower.logger import logging_framework

logger = logging_framework.Logger()

def detect_white_strict(img):
    """Detect white pixels with strict thresholds"""
    lower_white = np.array([250, 250, 250])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    return mask, np.sum(mask > 0) > 100

def detect_white_relaxed(img):
    """Detect white pixels with relaxed thresholds"""
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    return mask, np.sum(mask > 0) > 100

def detect_brightest_pixels(img):
    """Detect the brightest pixels in the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = np.percentile(gray, 90)  # Top 10% brightest pixels
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask, np.sum(mask > 0) > 100

def detect_adaptive_threshold(img):
    """Use adaptive thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    return mask, np.sum(mask > 0) > 100

def detect_color_edges(img):
    """Detect edges and assume they might be lines"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges to make them thicker
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=2)
    return mask, np.sum(mask > 0) > 100

def process_linreg(img, current_heading=0.0):
    """
    Simple line detection using OpenCV fitLine.
    Args:
        img: Input image (BGR format)
        current_heading: Current drone heading in radians for visualization
    Returns:
        tuple: (display_img, line_info) where line_info contains:
               {'x': x_position, 'y': y_position, 'vx': direction_x, 'vy': direction_y, 
                'slope': slope, 'intercept': intercept, 'is_vertical': bool, 'confidence': float}
               or None if no line detected
    """

    # Create working copy
    result_img = img.copy()
    
    # First, let's analyze the actual image content
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try multiple detection strategies
    detection_methods = [
        ("white_strict", lambda: detect_white_strict(img)),
        ("white_relaxed", lambda: detect_white_relaxed(img)),
        ("brightest_pixels", lambda: detect_brightest_pixels(img)),
        ("adaptive_threshold", lambda: detect_adaptive_threshold(img)),
        ("color_edges", lambda: detect_color_edges(img))
    ]
    
    for method_name, method_func in detection_methods:
        try:
            binary, valid = method_func()
            white_pixels = np.sum(binary > 0)
            if valid:
                break
        except Exception as e:
            continue
    else:
        
        result_color = img.copy()
        cv2.rectangle(result_color, (5, 5), (300, 80), (0, 0, 0), -1)  # Black background
        cv2.putText(result_color, "NO LINE DETECTED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_color, "All methods failed", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add center crosshair
        rows, cols = img.shape[:2]
        img_center_x = cols // 2
        img_center_y = rows // 2
        cv2.line(result_color, (img_center_x - 30, img_center_y), (img_center_x + 30, img_center_y), (255, 255, 255), 3)
        cv2.line(result_color, (img_center_x, img_center_y - 30), (img_center_x, img_center_y + 30), (255, 255, 255), 3)
        
        return result_color, None
    
    # Now we have a binary mask from one of the detection methods
    # Find line points for cv2.fitLine
    points = np.argwhere(binary > 0)
    
    if len(points) < 10:  # Need minimum points for reliable fitting
        return result_img, None
    
    # Convert points to (x,y) format for cv2.fitLine
    points = points[:, ::-1]  # swap (row,col) to (x,y)
    
    # Fit line using OpenCV's robust line fitting
    line_fit = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line_fit.flatten()  # extract scalars from numpy arrays
    
    # CHANGED: Ensure consistent direction - prefer NEGATIVE y direction (upward in image)
    # This matches the updated direction convention in detector.py
    if vy > 0:  # If pointing downward
        vx = -vx
        vy = -vy  # Flip to point upward
    
    # Calculate desired heading from line direction (same formula as in tracker.py)
    vx_bd, vy_bd = vx, -vy  # Approximation of camera-to-body transform for visualization
    desired_heading = np.arctan2(vx_bd, vy_bd)  # This matches tracker.py's calculation
    
    # Calculate angular error
    angular_error = normalize_angle(desired_heading - current_heading)
    
    # Calculate line parameters
    rows, cols = binary.shape
    
    # Calculate slope, intercept, and vertical flag
    if abs(vx) > 1e-6:  # Not vertical
        slope = vy / vx
        intercept = y0 - slope * x0
        is_vertical = False
    else:  # Vertical line
        slope = float('inf')
        intercept = x0  # For vertical lines, store x-intercept
        is_vertical = True
    
    # Calculate confidence based on number of points and line fit quality
    confidence = min(1.0, len(points) / 500.0)  # Normalize by expected max points
    
    # Calculate line endpoints for visualization
    lefty = int((-x0 * vy / vx) + y0) if vx != 0 else 0
    righty = int(((cols - x0) * vy / vx) + y0) if vx != 0 else rows
    
    # Create visualization on original image for better debugging
    result_color = img.copy()  # Use original image as base
    
    # Create a 3-channel version of binary for proper overlay
    if len(binary.shape) == 2:
        binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        binary_3ch = binary.copy()
    
    # Overlay detected line pixels in green for visibility
    line_overlay = np.zeros_like(result_color)
    # Set green pixels where line was detected
    mask_indices = binary > 0
    line_overlay[mask_indices] = [0, 255, 0]  # Green pixels where line detected
    
    # Blend the overlay with the original image
    result_color = cv2.addWeighted(result_color, 0.7, line_overlay, 0.3, 0)
    
    # Draw fitted line in bright red for high visibility
    if abs(vx) > 1e-6:  # Not vertical
        cv2.line(result_color, (cols - 1, righty), (0, lefty), (0, 0, 255), 5)  # Thick red line
    else:  # Vertical line
        cv2.line(result_color, (int(x0), 0), (int(x0), rows-1), (0, 0, 255), 5)  # Vertical red line
    
    # Add reference point and direction vector visualization
    center_x = int(x0)
    center_y = int(y0)
    cv2.circle(result_color, (center_x, center_y), 15, (255, 0, 255), -1)  # Magenta center point (larger)
    
    # Draw direction arrow - highlighting the ACTUAL direction that will be used
    arrow_length = 100
    end_x = int(center_x + arrow_length * vx)
    end_y = int(center_y + arrow_length * vy)
    cv2.arrowedLine(result_color, (center_x, center_y), (end_x, end_y), (0, 255, 255), 6)  # Cyan arrow (thicker)
    
    # Add text background for better readability
    cv2.rectangle(result_color, (5, 5), (400, 160), (0, 0, 0), -1)  # Black background
    
    
    
    # Calculate slope and intercept
    if abs(vx) > 1e-6:
        slope = vy / vx
        intercept = y0 - slope * x0
        is_vertical = False
    else:
        slope = np.inf
        intercept = np.nan
        is_vertical = True
    
    # Calculate confidence based on number of inlier points
    confidence = min(1.0, len(points) / 1000.0)  # Simple confidence metric
    
    # Add comprehensive debug info to visualization with better visibility
    cv2.putText(result_color, f"Confidence: {confidence:.3f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_color, f"Center: ({center_x}, {center_y})", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_color, f"Direction: ({vx:.3f}, {vy:.3f})", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_color, f"Target: ({target_x}, {target_y})", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_color, f"Points: {len(points)}", (10, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result_color, f"Method: {method_name}", (10, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add image center crosshair for reference (bright white)
    img_center_x = cols // 2
    img_center_y = rows // 2
    cv2.line(result_color, (img_center_x - 30, img_center_y), (img_center_x + 30, img_center_y), (255, 255, 255), 3)
    cv2.line(result_color, (img_center_x, img_center_y - 30), (img_center_x, img_center_y + 30), (255, 255, 255), 3)
    
    
    
    
    # Simulate basic control calculations for visualization
    EXTEND = 300
    IMAGE_WIDTH = img.shape[1] 
    IMAGE_HEIGHT = img.shape[0]
    CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2])
    
    # Calculate what the control system would calculate
    target_x = x0 + EXTEND * vx
    target_y = y0 + EXTEND * vy
    error_x = target_x - CENTER[0]
    error_y = target_y - CENTER[1]
    
    # Prepare line info (unchanged)
    line_info = {
        'method': method_name,
        'points_detected': len(points),
        'error_x': float(error_x),
        'error_y': float(error_y),
        'x_position': float(x0),
        'y_position': float(y0),
        'direction_x': float(vx),
        'direction_y': float(vy),
        'slope': float(slope),
        'intercept': float(intercept),
        'is_vertical': is_vertical,
        'confidence': confidence,
        'x': float(x0),
        'y': float(y0), 
        'vx': float(vx),
        'vy': float(vy)
    }
    
    return img, line_info


class linreg(Node):
    def __init__(self):
        super().__init__('linreg_test')
        # self.logger = get_logger('linreg')
        # self.logger.system("startup", "Simple Line Regression Node Started")
        
        # Configure QoS profile for PX4 messages
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Store current heading for visualization
        self.current_heading = 0.0
        self.vehicle_local_position = None

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.annotated_img_pub = self.create_publisher(Image, 'final_image', 10)
        
        # Subscribe to vehicle position to get current heading
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
            
    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position
        self.current_heading = self.get_current_heading()
        
    def get_current_heading(self):
        """
        Extract current heading (yaw) from vehicle local position quaternion.
        Returns heading in radians (-pi to pi).
        """
        if self.vehicle_local_position is None:
            return 0.0
            
        try:
            # Extract quaternion from vehicle local position
            q = [
                self.vehicle_local_position.q[1],  # x
                self.vehicle_local_position.q[2],  # y  
                self.vehicle_local_position.q[3],  # z
                self.vehicle_local_position.q[0]   # w (PX4 uses w-first format, tf uses w-last)
            ]
            
            # Convert quaternion to euler angles
            euler = tft.euler_from_quaternion(q)
            yaw = euler[2]  # Yaw is the third component (roll, pitch, yaw)
            
            return yaw
        except (AttributeError, IndexError):
            # Return 0 if quaternion data is not available yet
            return 0.0

    def image_callback(self, msg):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process line detection with current heading
            result_img, line_info = process_linreg(cv_image, self.current_heading)
            
            # Publish result image
            ros_image_msg = bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
            self.annotated_img_pub.publish(ros_image_msg)
        except:
            logger.log("linreg_error", logging_framework.LoggerColors.RED, "Failed to process image", 1000, force=True)


def main(args=None):
    rclpy.init(args=args)
    node = linreg()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()