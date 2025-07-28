#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

class LineFollowingVisualizer:
    """
    Comprehensive visualization framework for line following debug information.
    All visualizations are drawn in camera frame coordinates.
    """
    
    def __init__(self, image_width: int = 768, image_height: int = 576, extend_pixels: int = 300):
        self.image_width = image_width
        self.image_height = image_height
        self.extend_pixels = extend_pixels
        self.center = np.array([image_width // 2, image_height // 2])
        
        # Color scheme for different elements
        self.colors = {
            'detected_line': (0, 0, 255),      # Red - detected line
            'line_center': (0, 255, 0),        # Green - line center point
            'target_point': (255, 0, 255),     # Magenta - extrapolated target point
            'image_center': (255, 255, 255),   # White - image center crosshair
            'desired_direction': (0, 255, 255), # Cyan - desired movement direction
            'velocity_vector': (255, 165, 0),   # Orange - velocity command vector
            'error_lines': (128, 128, 128),     # Gray - error visualization lines
            'info_bg': (0, 0, 0),              # Black - text background
            'info_text': (255, 255, 255),      # White - text
            'target_path': (255, 255, 0),      # Yellow - path from center to target
        }
    
    def create_debug_visualization(self, 
                                 image: np.ndarray,
                                 line_params: Optional[Dict[str, float]] = None,
                                 velocity_commands: Optional[Dict[str, float]] = None,
                                 control_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create comprehensive debug visualization on the camera image.
        
        Args:
            image: Input camera image (grayscale or BGR)
            line_params: {'x': float, 'y': float, 'vx': float, 'vy': float}
            velocity_commands: {'vx_dc': float, 'vy_dc': float, 'wz_dc': float}
            control_info: Additional control information for display
            
        Returns:
            Annotated BGR image with all debug information
        """
        # Convert to BGR if needed
        if len(image.shape) == 2:
            result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result_img = image.copy()
        
        # Draw image center crosshair (always visible)
        self._draw_image_center(result_img)
        
        # Draw line-related information if available
        if line_params is not None:
            self._draw_line_detection(result_img, line_params)
            
            # Calculate and draw target point and path
            target_point = self._calculate_target_point(line_params)
            self._draw_target_point_and_path(result_img, line_params, target_point)
            
            # Draw control errors
            self._draw_control_errors(result_img, target_point)
        
        # Draw velocity commands if available
        if velocity_commands is not None:
            self._draw_velocity_commands(result_img, velocity_commands)
        
        # Draw information panel
        self._draw_info_panel(result_img, line_params, velocity_commands, control_info)
        
        return result_img
    
    def _draw_image_center(self, img: np.ndarray):
        """Draw crosshair at image center"""
        center_x, center_y = self.center
        # Horizontal line
        cv2.line(img, (center_x - 30, center_y), (center_x + 30, center_y), 
                self.colors['image_center'], 3)
        # Vertical line  
        cv2.line(img, (center_x, center_y - 30), (center_x, center_y + 30), 
                self.colors['image_center'], 3)
        # Center dot
        cv2.circle(img, (center_x, center_y), 5, self.colors['image_center'], -1)
    
    def _draw_line_detection(self, img: np.ndarray, line_params: Dict[str, float]):
        """Draw the detected line and its center point"""
        x, y, vx, vy = line_params['x'], line_params['y'], line_params['vx'], line_params['vy']
        
        # Draw line center point (larger for visibility)
        cv2.circle(img, (int(x), int(y)), 8, self.colors['line_center'], -1)
        cv2.circle(img, (int(x), int(y)), 12, self.colors['line_center'], 2)
        
        # Draw detected line extending in both directions
        line_length = 200
        pt1 = (int(x - line_length * vx), int(y - line_length * vy))
        pt2 = (int(x + line_length * vx), int(y + line_length * vy))
        cv2.line(img, pt1, pt2, self.colors['detected_line'], 4)
        
        # Draw direction arrow on the line
        arrow_start = (int(x), int(y))
        arrow_end = (int(x + 50 * vx), int(y + 50 * vy))
        cv2.arrowedLine(img, arrow_start, arrow_end, self.colors['detected_line'], 3, tipLength=0.3)
    
    def _calculate_target_point(self, line_params: Dict[str, float]) -> Tuple[float, float]:
        """Calculate the extrapolated target point"""
        x, y, vx, vy = line_params['x'], line_params['y'], line_params['vx'], line_params['vy']
        target_x = x + self.extend_pixels * vx
        target_y = y + self.extend_pixels * vy
        return (target_x, target_y)
    
    def _draw_target_point_and_path(self, img: np.ndarray, line_params: Dict[str, float], target_point: Tuple[float, float]):
        """Draw the target point and path from center to target"""
        target_x, target_y = target_point
        center_x, center_y = self.center
        
        # Draw target point (large, highly visible)
        cv2.circle(img, (int(target_x), int(target_y)), 12, self.colors['target_point'], -1)
        cv2.circle(img, (int(target_x), int(target_y)), 18, self.colors['target_point'], 3)
        
        # Draw path from image center to target point
        cv2.line(img, (center_x, center_y), (int(target_x), int(target_y)), 
                self.colors['target_path'], 3)
        
        # Draw target point label
        cv2.putText(img, "TARGET", (int(target_x) + 25, int(target_y) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['target_point'], 2)
    
    def _draw_control_errors(self, img: np.ndarray, target_point: Tuple[float, float]):
        """Draw control error visualization"""
        target_x, target_y = target_point
        center_x, center_y = self.center
        
        # Calculate errors
        error_x = target_x - center_x
        error_y = target_y - center_y
        
        # Draw error components as lines
        # X error (horizontal)
        if abs(error_x) > 5:  # Only draw if significant
            error_end_x = center_x + error_x
            cv2.line(img, (center_x, center_y), (int(error_end_x), center_y), 
                    self.colors['error_lines'], 2)
            cv2.putText(img, f"X_err: {error_x:.0f}", (center_x + 10, center_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['error_lines'], 1)
        
        # Y error (vertical)
        if abs(error_y) > 5:  # Only draw if significant
            error_end_y = center_y + error_y
            cv2.line(img, (center_x, center_y), (center_x, int(error_end_y)), 
                    self.colors['error_lines'], 2)
            cv2.putText(img, f"Y_err: {error_y:.0f}", (center_x + 15, center_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['error_lines'], 1)
    
    def _draw_velocity_commands(self, img: np.ndarray, velocity_commands: Dict[str, float]):
        """Draw velocity command vectors in camera frame"""
        center_x, center_y = self.center
        vx_dc = velocity_commands.get('vx_dc', 0.0)
        vy_dc = velocity_commands.get('vy_dc', 0.0)
        wz_dc = velocity_commands.get('wz_dc', 0.0)
        
        # Scale velocity for visualization (adjust scale factor as needed)
        scale_factor = 100.0
        vel_end_x = center_x + vx_dc * scale_factor
        vel_end_y = center_y + vy_dc * scale_factor
        
        # Draw velocity vector arrow
        if abs(vx_dc) > 0.01 or abs(vy_dc) > 0.01:  # Only draw if significant
            cv2.arrowedLine(img, (center_x, center_y), (int(vel_end_x), int(vel_end_y)), 
                           self.colors['velocity_vector'], 4, tipLength=0.2)
            
            # Label the velocity vector
            cv2.putText(img, "VEL_CMD", (int(vel_end_x) + 10, int(vel_end_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['velocity_vector'], 2)
        
        # Draw yaw command visualization (small arc or indicator)
        if abs(wz_dc) > 0.01:
            self._draw_yaw_indicator(img, center_x, center_y, wz_dc)
    
    def _draw_yaw_indicator(self, img: np.ndarray, center_x: int, center_y: int, wz_dc: float):
        """Draw yaw rate command as a circular arc indicator"""
        radius = 40
        # Scale yaw rate for visualization
        angle_span = min(abs(wz_dc) * 50, 90)  # Limit to 90 degrees max
        
        if wz_dc > 0:  # Counterclockwise
            start_angle = -10
            end_angle = start_angle + angle_span
            color = (0, 255, 0)  # Green for positive yaw
        else:  # Clockwise
            start_angle = 10
            end_angle = start_angle - angle_span
            color = (0, 0, 255)  # Red for negative yaw
        
        # Draw arc
        cv2.ellipse(img, (center_x, center_y), (radius, radius), 0, 
                   start_angle, end_angle, color, 3)
        
        # Draw arrow at end of arc
        end_angle_rad = np.radians(end_angle)
        arrow_x = center_x + radius * np.cos(end_angle_rad)
        arrow_y = center_y + radius * np.sin(end_angle_rad)
        cv2.circle(img, (int(arrow_x), int(arrow_y)), 5, color, -1)
    
    def _draw_info_panel(self, img: np.ndarray, 
                        line_params: Optional[Dict[str, float]] = None,
                        velocity_commands: Optional[Dict[str, float]] = None,
                        control_info: Optional[Dict[str, Any]] = None):
        """Draw information panel with all numeric data"""
        # Info panel background
        panel_height = 200
        panel_width = 400
        alpha = 0.3
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), self.colors['info_bg'], -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        y_offset = 25
        line_height = 25
        
        # Line parameters
        if line_params is not None:
            cv2.putText(img, f"LINE: x={line_params['x']:.1f}, y={line_params['y']:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info_text'], 1)
            y_offset += line_height
            
            cv2.putText(img, f"DIR:  vx={line_params['vx']:.3f}, vy={line_params['vy']:.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info_text'], 1)
            y_offset += line_height
            
            # Calculate and show target point
            target_x, target_y = self._calculate_target_point(line_params)
            cv2.putText(img, f"TARGET: ({target_x:.1f}, {target_y:.1f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['target_point'], 1)
            y_offset += line_height
        
        # Velocity commands
        if velocity_commands is not None:
            cv2.putText(img, f"VEL_DC: vx={velocity_commands.get('vx_dc', 0):.3f}, vy={velocity_commands.get('vy_dc', 0):.3f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['velocity_vector'], 1)
            y_offset += line_height
            
            cv2.putText(img, f"YAW_DC: wz={velocity_commands.get('wz_dc', 0):.3f} rad/s", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['velocity_vector'], 1)
            y_offset += line_height
        
        # Additional control info
        if control_info is not None:
            for key, value in control_info.items():
                if isinstance(value, (int, float)):
                    cv2.putText(img, f"{key}: {value:.3f}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info_text'], 1)
                else:
                    cv2.putText(img, f"{key}: {value}", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info_text'], 1)
                y_offset += line_height
                if y_offset > panel_height - 20:  # Prevent overflow
                    break


# Convenience function for easy integration
def create_line_following_debug_image(image: np.ndarray,
                                    line_x: float = None, line_y: float = None, 
                                    line_vx: float = None, line_vy: float = None,
                                    vx_dc: float = None, vy_dc: float = None, wz_dc: float = None,
                                    **additional_info) -> np.ndarray:
    """
    Convenience function to create debug visualization with minimal setup.
    
    Usage example:
        debug_img = create_line_following_debug_image(
            image=camera_image,
            line_x=383.5, line_y=287.5, line_vx=1.0, line_vy=0.0,
            vx_dc=0.5, vy_dc=-0.7, wz_dc=0.1,
            error_x=50, error_y=-25
        )
    """
    visualizer = LineFollowingVisualizer()
    
    # Package parameters
    line_params = None
    if all(v is not None for v in [line_x, line_y, line_vx, line_vy]):
        line_params = {'x': line_x, 'y': line_y, 'vx': line_vx, 'vy': line_vy}
    
    velocity_commands = None
    if any(v is not None for v in [vx_dc, vy_dc, wz_dc]):
        velocity_commands = {
            'vx_dc': vx_dc or 0.0,
            'vy_dc': vy_dc or 0.0, 
            'wz_dc': wz_dc or 0.0
        }
    
    return visualizer.create_debug_visualization(
        image=image,
        line_params=line_params,
        velocity_commands=None,
        control_info=additional_info if additional_info else None
    )