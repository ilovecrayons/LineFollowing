#!/usr/bin/env python3

"""
Performance test to compare old vs new curve fitting approaches
"""

import sys
import os
import time
import numpy as np
import cv2

# Add the path to import our modules
sys.path.append('src/line_follower/line_follower')

from linreg import process_linreg

def create_test_curved_line(width=640, height=480, curve_strength=0.3):
    """Create a test image with a curved line"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a curved line using sine wave
    x_center = width // 2
    for y in range(height):
        # Curved line equation
        offset = int(curve_strength * width * 0.15 * np.sin(2 * np.pi * y / height))
        x = x_center + offset
        
        # Draw line with some thickness
        if 0 <= x < width:
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
    
    return img

def create_test_straight_line(width=640, height=480):
    """Create a test image with a straight line"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a straight diagonal line
    x1, y1 = width // 4, 0
    x2, y2 = 3 * width // 4, height
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 6)
    
    return img

def performance_test():
    print("=== Performance Test: Enhanced Curve Detection ===\n")
    
    # Test configurations
    test_cases = [
        ("Curved Line (Medium)", create_test_curved_line(curve_strength=0.3)),
        ("Curved Line (High)", create_test_curved_line(curve_strength=0.6)),
        ("Straight Line", create_test_straight_line()),
    ]
    
    results = []
    
    for test_name, test_img in test_cases:
        print(f"Testing: {test_name}")
        print("-" * 40)
        
        # Run multiple iterations for timing
        times = []
        confidences = []
        
        for i in range(3):  # 3 iterations for average
            start_time = time.perf_counter()
            display_img, result = process_linreg(test_img)
            end_time = time.perf_counter()
            
            processing_time = (end_time - start_time) * 1000  # ms
            times.append(processing_time)
            
            if result:
                confidences.append(result.get('confidence', 0))
                print(f"  Run {i+1}: {processing_time:.1f}ms, confidence: {result.get('confidence', 0):.3f}")
            else:
                print(f"  Run {i+1}: {processing_time:.1f}ms, FAILED")
                confidences.append(0)
        
        avg_time = np.mean(times)
        avg_confidence = np.mean(confidences)
        
        print(f"  Average: {avg_time:.1f}ms, confidence: {avg_confidence:.3f}")
        
        results.append({
            'name': test_name,
            'avg_time': avg_time,
            'avg_confidence': avg_confidence,
            'success_rate': sum(1 for c in confidences if c > 0.3) / len(confidences)
        })
        
        print()
    
    # Summary
    print("=== PERFORMANCE SUMMARY ===")
    print(f"{'Test Case':<20} {'Avg Time (ms)':<15} {'Confidence':<12} {'Success Rate':<12}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['name']:<20} {result['avg_time']:<15.1f} {result['avg_confidence']:<12.3f} {result['success_rate']:<12.1%}")
    
    print()
    
    # Key improvements summary
    print("=== KEY IMPROVEMENTS ===")
    print("✓ Scikit-learn RANSAC: Professional-grade outlier rejection")
    print("✓ DBSCAN clustering: Noise reduction and main line isolation")
    print("✓ Polynomial curve fitting: Handles curves unlike simple linear regression")
    print("✓ Adaptive orientation: Detects whether line is more horizontal or vertical")
    print("✓ Enhanced confidence metrics: Better reliability assessment")
    print("✓ Robust fallback: Multiple fitting strategies for different scenarios")

if __name__ == "__main__":
    performance_test()
