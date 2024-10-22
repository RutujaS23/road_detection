import cv2
import numpy as np
import os
import random
import time

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

def lane_detection(image_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to read the image.")
            return

        height, width = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Sobel operator for edge detection
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.uint8(edges)

        # Apply thresholding to edges
        _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Define the region of interest
        vertices = np.array([[0, height], [width / 2, height / 2], [width, height]], dtype=np.int32)
        roi = region_of_interest(thresh, [vertices])

        # Hough Transform
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        # Draw the lines on the original image
        if lines is not None:
            draw_lines(img, lines)

        # Show the output
        cv2.imshow('Lane Detection', img)
        cv2.waitKey(1)
        time.sleep(3)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", str(e))

# Select 10 random images from the img folder
img_folder = 'img'
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]
random_imgs = random.sample(img_files, 1)

for img in random_imgs:
    img_path = os.path.join(img_folder, img)
    print("Selected Image:", img_path)
    lane_detection(img_path)