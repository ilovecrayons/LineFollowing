#!/usr/bin/env python3
"""
Test script to verify the enhanced curve-aware line detection
"""
import numpy as np
import cv2
import sys
import os

# Add the source path to test the linreg module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'line_follower', 'line_follower'))

from linreg import process_linreg

def create_test_image_curved():
    """Create a test image with a curved line"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a curved line using quadratic function
    center_x = 320
    for y in range(480):
        # Quadratic curve: x = center + 0.001 * (y - 240)^2
        x_offset = 0.0005 * (y - 240)**2
        x = int(center_x + x_offset)
        
        if 0 <= x < 640:
            # Draw thick line
            for dx in range(-8, 9):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < 640 and 0 <= py < 480:
                        img[py, px] = [255, 255, 255]  # White line
    
    return img

def create_test_image_straight():
    """Create a test image with a straight line"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a straight diagonal line
    for y in range(480):
        x = int(320 + 0.3 * (y - 240))  # Slight diagonal
        
        if 0 <= x < 640:
            # Draw thick line
            for dx in range(-8, 9):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < 640 and 0 <= py < 480:
                        img[py, px] = [255, 255, 255]  # White line
    
    return img

def test_curve_detection():
    """Test the curve detection on both straight and curved lines"""
    print("Testing Enhanced Curve-Aware Line Detection")
    print("=" * 50)
    
    # Test curved line
    print("\n1. Testing curved line:")
    curved_img = create_test_image_curved()
    
    try:
        result_img, line_info = process_linreg(curved_img)
        
        if line_info is not None:
            print(f"   ✓ Curve detected successfully!")
            print(f"   Position: ({line_info.get('x_position', 'N/A'):.1f}, {line_info.get('y_position', 'N/A'):.1f})")
            print(f"   Direction: ({line_info.get('direction_x', 'N/A'):.3f}, {line_info.get('direction_y', 'N/A'):.3f})")
            print(f"   Confidence: {line_info.get('confidence', 'N/A'):.3f}")
            print(f"   Is vertical: {line_info.get('is_vertical', 'N/A')}")
            
            # Save result
            cv2.imwrite('/home/huang/LineFollowing/test_curved_result.jpg', result_img)
            print(f"   Result saved to test_curved_result.jpg")
        else:
            print("   ✗ Failed to detect curved line")
            
    except Exception as e:
        print(f"   ✗ Error processing curved line: {e}")
    
    # Test straight line
    print("\n2. Testing straight line:")
    straight_img = create_test_image_straight()
    
    try:
        result_img, line_info = process_linreg(straight_img)
        
        if line_info is not None:
            print(f"   ✓ Straight line detected successfully!")
            print(f"   Position: ({line_info.get('x_position', 'N/A'):.1f}, {line_info.get('y_position', 'N/A'):.1f})")
            print(f"   Direction: ({line_info.get('direction_x', 'N/A'):.3f}, {line_info.get('direction_y', 'N/A'):.3f})")
            print(f"   Confidence: {line_info.get('confidence', 'N/A'):.3f}")
            print(f"   Is vertical: {line_info.get('is_vertical', 'N/A')}")
            
            # Save result
            cv2.imwrite('/home/huang/LineFollowing/test_straight_result.jpg', result_img)
            print(f"   Result saved to test_straight_result.jpg")
        else:
            print("   ✗ Failed to detect straight line")
            
    except Exception as e:
        print(f"   ✗ Error processing straight line: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_curve_detection()
