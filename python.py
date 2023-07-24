import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect_lanes(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define the region of interest (ROI)
    height, width = img.shape[0], img.shape[1]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=5)
    
    # Draw the detected lines on a copy of the original image
    line_image = np.copy(img) * 0
    draw_lines(line_image, lines)
    lane_lines = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    
    return lane_lines

# Read the input image
input_image = cv2.imread('path_to_image.jpg')

# Detect lanes
result_image = detect_lanes(input_image)

# Display the result
cv2.imshow('Lane Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

