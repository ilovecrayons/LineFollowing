#!/usr/bin/env python3

"""
Comprehensive Performance and Accuracy Test for Line Detection Algorithm
Target: >10 FPS (< 100ms per frame) with high accuracy
"""

import sys
import os
import time
import numpy as np
import cv2
import statistics

# Add the path to import our modules
sys.path.append('src/line_follower/line_follower')

from linreg import process_linreg

def create_test_scenarios():
    """Create various test scenarios to validate algorithm performance"""
    scenarios = []
    
    # 1. Simple straight line (baseline)
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(img1, (300, 0), (340, 479), (255, 255, 255), 8)
    scenarios.append(("Straight Line", img1))
    
    # 2. Curved line (medium curve)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        x = int(320 + 80 * np.sin(2 * np.pi * y / 480))
        if 0 <= x < 640:
            cv2.circle(img2, (x, y), 4, (255, 255, 255), -1)
    scenarios.append(("Medium Curve", img2))
    
    # 3. Sharp curve (stress test)
    img3 = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        x = int(320 + 120 * np.sin(3 * np.pi * y / 480))
        if 0 <= x < 640:
            cv2.circle(img3, (x, y), 4, (255, 255, 255), -1)
    scenarios.append(("Sharp Curve", img3))
    
    # 4. Noisy line (real-world simulation)
    img4 = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        x = int(320 + 0.2 * (y - 240))
        if 0 <= x < 640:
            cv2.circle(img4, (x, y), 3, (255, 255, 255), -1)
    # Add noise
    noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    img4 = cv2.add(img4, noise)
    scenarios.append(("Noisy Line", img4))
    
    # 5. Broken line (challenging case)
    img5 = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(0, 480, 15):  # Every 15 pixels
        if y % 30 < 10:  # Draw for 10 pixels, skip for 20
            for dy in range(10):
                if y + dy < 480:
                    x = int(320 + 0.1 * (y + dy - 240))
                    if 0 <= x < 640:
                        cv2.circle(img5, (x, y + dy), 3, (255, 255, 255), -1)
    scenarios.append(("Broken Line", img5))
    
    return scenarios

def performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("🚀 LINE DETECTION PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Target: >10 FPS (< 100ms per frame)")
    print()
    
    scenarios = create_test_scenarios()
    all_results = []
    
    for scenario_name, test_img in scenarios:
        print(f"📊 Testing: {scenario_name}")
        print("-" * 40)
        
        times = []
        confidences = []
        success_count = 0
        
        # Warm up run (exclude from timing)
        try:
            process_linreg(test_img)
        except:
            pass
        
        # Performance test runs
        for run in range(10):  # 10 runs for statistical significance
            start_time = time.perf_counter()
            
            try:
                display_img, result = process_linreg(test_img)
                end_time = time.perf_counter()
                
                processing_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(processing_time)
                
                if result and 'confidence' in result:
                    confidence = result['confidence']
                    confidences.append(confidence)
                    if confidence > 0.3:  # Minimum acceptable confidence
                        success_count += 1
                    
                    if run < 3:  # Only print first 3 runs to reduce output
                        print(f"  Run {run+1}: {processing_time:.1f}ms, conf: {confidence:.3f}")
                else:
                    confidences.append(0.0)
                    if run < 3:
                        print(f"  Run {run+1}: {processing_time:.1f}ms, FAILED")
                        
            except Exception as e:
                end_time = time.perf_counter()
                processing_time = (end_time - start_time) * 1000
                times.append(processing_time)
                confidences.append(0.0)
                if run < 3:
                    print(f"  Run {run+1}: {processing_time:.1f}ms, ERROR: {e}")
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        avg_confidence = statistics.mean(confidences)
        success_rate = success_count / len(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        # Performance assessment
        fps_status = "✅ EXCELLENT" if fps >= 15 else "✅ GOOD" if fps >= 10 else "⚠️  SLOW" if fps >= 5 else "❌ TOO SLOW"
        confidence_status = "✅ HIGH" if avg_confidence >= 0.7 else "✅ GOOD" if avg_confidence >= 0.5 else "⚠️  LOW" if avg_confidence >= 0.3 else "❌ POOR"
        
        print(f"  📈 Statistics:")
        print(f"    • Average: {avg_time:.1f}ms ± {std_time:.1f}ms")
        print(f"    • Range: {min_time:.1f}ms - {max_time:.1f}ms")
        print(f"    • FPS: {fps:.1f} {fps_status}")
        print(f"    • Confidence: {avg_confidence:.3f} {confidence_status}")
        print(f"    • Success Rate: {success_rate:.1%}")
        print()
        
        all_results.append({
            'name': scenario_name,
            'avg_time': avg_time,
            'fps': fps,
            'confidence': avg_confidence,
            'success_rate': success_rate,
            'std_time': std_time
        })
    
    # Overall summary
    print("🎯 OVERALL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<15} {'Avg Time (ms)':<15} {'FPS':<8} {'Confidence':<12} {'Success Rate':<12}")
    print("-" * 70)
    
    total_fps = 0
    for result in all_results:
        fps_indicator = "🟢" if result['fps'] >= 10 else "🟡" if result['fps'] >= 5 else "🔴"
        print(f"{result['name']:<15} {result['avg_time']:<15.1f} {result['fps']:<8.1f} {result['confidence']:<12.3f} {result['success_rate']:<12.1%} {fps_indicator}")
        total_fps += result['fps']
    
    avg_fps = total_fps / len(all_results)
    print("-" * 70)
    print(f"{'AVERAGE':<15} {'':<15} {avg_fps:<8.1f}")
    print()
    
    # Performance recommendations
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    if avg_fps < 10:
        print("❌ CRITICAL: Average FPS below target (10 FPS)")
        print("   Recommendations:")
        print("   • Reduce RANSAC_MAX_TRIALS (currently 100)")
        print("   • Disable DBSCAN for small point sets")
        print("   • Use simpler polynomial degree (degree=1)")
        print("   • Implement early exit conditions")
    elif avg_fps < 15:
        print("⚠️  WARNING: FPS acceptable but could be improved")
        print("   Recommendations:")
        print("   • Fine-tune RANSAC parameters")
        print("   • Optimize DBSCAN parameters")
        print("   • Consider caching frequent calculations")
    else:
        print("✅ EXCELLENT: Performance exceeds requirements!")
        print("   Current optimizations are working well.")
    
    print()
    slowest = min(all_results, key=lambda x: x['fps'])
    if slowest['fps'] < 10:
        print(f"🐌 Slowest scenario: {slowest['name']} ({slowest['fps']:.1f} FPS)")
        print("   This scenario needs specific optimization attention.")
    
    return all_results

def accuracy_test():
    """Test accuracy with known ground truth"""
    print("🎯 ACCURACY VALIDATION TEST")
    print("=" * 60)
    
    # Test with known straight line
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw perfect vertical line at x=320
    cv2.line(test_img, (320, 0), (320, 479), (255, 255, 255), 6)
    
    print("Testing perfect vertical line at x=320...")
    _, result = process_linreg(test_img)
    
    if result:
        detected_x = result.get('x_position', 0)
        error = abs(detected_x - 320)
        print(f"✅ Detected position: x={detected_x:.1f}")
        print(f"   Ground truth: x=320.0")
        print(f"   Error: {error:.1f} pixels")
        print(f"   Accuracy: {'EXCELLENT' if error < 5 else 'GOOD' if error < 10 else 'POOR'}")
    else:
        print("❌ Failed to detect line")
    
    print()

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE LINE DETECTION ANALYSIS")
    print("Date:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Target Performance: >10 FPS for drone real-time control")
    print()
    
    # Run accuracy test first
    accuracy_test()
    
    # Run performance benchmark
    results = performance_benchmark()
    
    # Final assessment
    avg_fps = sum(r['fps'] for r in results) / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    
    print("🏁 FINAL ASSESSMENT")
    print("=" * 60)
    
    if avg_fps >= 10 and avg_confidence >= 0.5:
        print("🎉 SUCCESS: Algorithm meets requirements!")
        print(f"   • Average FPS: {avg_fps:.1f} (target: >10)")
        print(f"   • Average Confidence: {avg_confidence:.3f} (target: >0.5)")
        print("   • Ready for real-time drone deployment")
    elif avg_fps >= 10:
        print("⚠️  PARTIAL SUCCESS: Good performance, but accuracy needs improvement")
        print(f"   • FPS: {avg_fps:.1f} ✅")
        print(f"   • Confidence: {avg_confidence:.3f} ⚠️")
    elif avg_confidence >= 0.5:
        print("⚠️  PARTIAL SUCCESS: Good accuracy, but performance needs optimization")
        print(f"   • FPS: {avg_fps:.1f} ⚠️")
        print(f"   • Confidence: {avg_confidence:.3f} ✅")
    else:
        print("❌ NEEDS IMPROVEMENT: Both performance and accuracy need work")
        print(f"   • FPS: {avg_fps:.1f} ❌")
        print(f"   • Confidence: {avg_confidence:.3f} ❌")
