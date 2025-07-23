from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from matplotlib.lines import Line2D
import time

# Further reduce minimum area for more sensitive detection
AREA_MIN      = 50  # Further reduced from 100 to 50
AR_MIN        = 2.5    
X_CLUSTER_PX  = 10       
VERTICAL_DEG  = 75.0   
VAR_RATIO     = 0.5      
SLOPE_LIMIT = np.tan(np.deg2rad(VERTICAL_DEG))


def calculate_regression(points, slope_limit=SLOPE_LIMIT, var_ratio=VAR_RATIO):
    if points.shape[0] < 2:
        return np.nan, np.nan, False, None
    
    x = points[:, 1].astype(np.float64)
    y = points[:, 0].astype(np.float64)
    
    vx = x.var()
    vy = y.var()
    
    if vy > 0 and vx < var_ratio * vy:
        return np.nan, np.nan, True, float(np.median(x))
    
    x_mean = x.mean()
    y_mean = y.mean()
    d = ((x - x_mean) ** 2).sum()
    if d == 0:
        return np.nan, np.nan, True, x_mean
    
    m = ((x - x_mean) * (y - y_mean)).sum() / d
    b = y_mean - m * x_mean
    
    if abs(m) > slope_limit:
        return np.nan, np.nan, True, float(np.median(x))
    
    return m, b, False, None

def find_inliers(m, b, shape, vertical=False, x0=None):
    h, w = shape
    if vertical:
        x = int(np.clip(round(x0), 0, w - 1)) # type: ignore
        return x, 0, x, h - 1
    
    x1 = 0
    y1 = m * x1 + b
    x2 = w - 1
    y2 = m * x2 + b
    
    y1 = int(np.clip(round(y1), 0, h - 1))
    y2 = int(np.clip(round(y2), 0, h - 1))
    return x1, y1, x2, y2

def _large_contour_info(contours, area_min=AREA_MIN):
    infos = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = h / max(w, 1)  
        xc = x + w / 2.0
        yc = y + h / 2.0
        vert_score = ar * area
        infos.append(dict(
            cnt=cnt, area=area, x=x, y=y, w=w, h=h,
            xc=xc, yc=yc, ar=ar, vert_score=vert_score
        ))
    return infos

def _cluster_by_x(infos, tol=X_CLUSTER_PX):
    if not infos:
        return []
    infos_sorted = sorted(infos, key=lambda d: d['xc'])
    clusters = []
    current = [infos_sorted[0]]
    last_x = infos_sorted[0]['xc']
    for d in infos_sorted[1:]:
        if abs(d['xc'] - last_x) <= tol:
            current.append(d)
        else:
            clusters.append(current)
            current = [d]
        last_x = d['xc']
    clusters.append(current)
    return clusters

def _cluster_metrics(cluster):
    areas = [d['area'] for d in cluster]
    ars   = [d['ar'] for d in cluster]
    xcs   = [d['xc'] for d in cluster]
    xs    = [d['x'] for d in cluster]
    ys    = [d['y'] for d in cluster]
    ws    = [d['w'] for d in cluster]
    hs    = [d['h'] for d in cluster]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max([x+w for x,w in zip(xs,ws)])
    y_max = max([y+h for y,h in zip(ys,hs)])
    w_tot = x_max - x_min
    h_tot = y_max - y_min
    ar_tot = h_tot / max(w_tot, 1)

    vert_score = sum(d['vert_score'] for d in cluster)

    return dict(
        contours=[d['cnt'] for d in cluster],
        area=sum(areas),
        ar=ar_tot,
        x=x_min, y=y_min, w=w_tot, h=h_tot,
        xc=float(np.median(xcs)),
        vert_score=vert_score
    )

def select_best_vertical_band(contours, area_min=AREA_MIN, ar_min=AR_MIN, merge_tol=X_CLUSTER_PX, thresh=None):
    cluster_start = time.perf_counter()
    
    info_start = time.perf_counter()
    infos = _large_contour_info(contours, area_min=area_min)
    info_time = time.perf_counter() - info_start
    #print(f"    [CLUSTER] Contour info extraction took: {info_time*1000:.3f}ms (processed {len(infos)} valid contours)")
    
    if not infos:
        return None, False, None, None, False

    group_start = time.perf_counter()
    clusters = _cluster_by_x(infos, tol=merge_tol)
    cluster_infos = [_cluster_metrics(c) for c in clusters]
    group_time = time.perf_counter() - group_start
    #print(f"    [CLUSTER] Grouping and metrics took: {group_time*1000:.3f}ms (created {len(clusters)} clusters)")

    best_start = time.perf_counter()
    best = max(cluster_infos, key=lambda d: d['vert_score'])

    mask = np.zeros_like(thresh, dtype=np.uint8)  
    cv2.drawContours(mask, best['contours'], -1, 255, thickness=-1) # type: ignore

    is_vertical = (best['ar'] >= ar_min) # type: ignore
    best_time = time.perf_counter() - best_start
    #print(f"    [CLUSTER] Best selection and mask creation took: {best_time*1000:.3f}ms")

    total_cluster = time.perf_counter() - cluster_start
    #print(f"    [CLUSTER] Total clustering took: {total_cluster*1000:.3f}ms")

    return mask, is_vertical, best['xc'], best, True


def process_linreg(img):
    process_start = time.perf_counter()  # More precise timer
    #print(f"  [LINREG] Starting linear regression processing (image shape: {img.shape})")
    
    # Keep all original processing logic, just remove matplotlib
    gray_start = time.perf_counter()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_time = time.perf_counter() - gray_start
    #print(f"  [LINREG] Color conversion took: {gray_time*1000:.3f}ms")

    morph_start = time.perf_counter()
    kernel_dilation = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(imgray, kernel_dilation, iterations=4)
    
    kernel_blur = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dilation, -1, kernel_blur)
    
    # Enhanced image processing with multiple methods
    # Very low threshold to catch any white lines
    _, thresh1 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    
    # Adaptive thresholding to handle varying lighting
    thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Edge detection to find line boundaries
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine all methods
    thresh = cv2.bitwise_or(cv2.bitwise_or(thresh1, thresh2), edges_dilated)
    
    morph_time = time.perf_counter() - morph_start
    #print(f"  [LINREG] Morphological operations took: {morph_time*1000:.3f}ms")
    
    contour_start = time.perf_counter()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_time = time.perf_counter() - contour_start
    print(f"  [LINREG] Contour finding took: {contour_time*1000:.3f}ms (found {len(contours)} contours)")
    
    cluster_start = time.perf_counter()
    mask, cluster_vertical, xc, cluster_info, found_big = select_best_vertical_band(
        contours, area_min=AREA_MIN, ar_min=AR_MIN, merge_tol=X_CLUSTER_PX, thresh=thresh
    )
    cluster_time = time.perf_counter() - cluster_start
    print(f"  [LINREG] Clustering took: {cluster_time*1000:.3f}ms")
    
    regression_start = time.perf_counter()
    if found_big:
        pts_src = mask
    else:
        pts_src = thresh  
    
    pts = np.column_stack(np.nonzero(pts_src)) # type: ignore
    
    force_vertical = cluster_vertical and found_big
    
    if force_vertical:
        m = b = np.nan
        vertical = True
        x0 = xc
    else:
        m, b, vertical, x0 = calculate_regression(pts)
    regression_time = time.perf_counter() - regression_start
    #print(f"  [LINREG] Regression calculation took: {regression_time*1000:.3f}ms (processed {len(pts)} points)")
    
    # Create display image with original processing results
    draw_start = time.perf_counter()
    display_img = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < AREA_MIN:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    
    if found_big and cluster_info is not None:
        x = int(cluster_info['x']); y = int(cluster_info['y']) # type: ignore
        w = int(cluster_info['w']); h = int(cluster_info['h']) # type: ignore
        cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 255), 5)
    
    # Draw the regression line using OpenCV instead of matplotlib
    if vertical:
        x = int(np.clip(round(x0), 0, imgray.shape[1] - 1)) # type: ignore
        cv2.line(display_img, (x, 0), (x, imgray.shape[0] - 1), (0, 255, 0), 5)
    elif not np.isnan(m) and not np.isnan(b):
        x1, y1, x2, y2 = find_inliers(m, b, imgray.shape, vertical=False)
        cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    draw_time = time.perf_counter() - draw_start
    print(f"  [LINREG] Drawing took: {draw_time*1000:.3f}ms")
    
    total_time = time.perf_counter() - process_start
    print(f"  [LINREG] Total processing took: {total_time*1000:.3f}ms")

    if vertical:
        x = int(np.clip(round(x0), 0, imgray.shape[1] - 1)) # type: ignore
        cv2.line(display_img, (x, 0), (x, imgray.shape[0] - 1), (0, 255, 0), 5)
        # Store vertical line endpoints
        line_points = (x, 0, x, imgray.shape[0] - 1)
    elif not np.isnan(m) and not np.isnan(b):
        x1, y1, x2, y2 = find_inliers(m, b, imgray.shape, vertical=False)
        cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Store normal line endpoints
        line_points = (x1, y1, x2, y2)
    else:
        line_points = None

    line_info = {
        'slope': m if not vertical else np.nan,
        'intercept': b if not vertical else np.nan,
        'is_vertical': vertical,
        'x_position': x0 if vertical else np.nan,
        'points': line_points
    }

    return display_img, line_info