#!/usr/bin/env python
# coding: utf-8

import argparse
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------#
def get_dot_numpy(image,scaling):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    
    corners = cv2.goodFeaturesToTrack(blur,676,0.01,5)
    
    corners = np.int16(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners_subpix = cv2.cornerSubPix(blur, np.float32(corners), (5, 5), (-1, -1), criteria)
    
    return corners_subpix * scaling
#-------------------------------------------------------------------------------------------------------------------#
def get_dot_centers(image_file, output_file_name, output_format, scaling):
# Load the image and convert it to grayscale
  image = cv2.imread(image_file)
  print(image.shape)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Blur the image to reduce high frequency noise
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  # Use the Hough transform to detect circles in the image
  circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=50)

  # Convert the circles object to a NumPy array
  try:
    circles = np.int16(circles)
  except:
    print("Dots width too small, try different format.")
    return 
  
  circles = circles[:,:,:2]
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
  corners_subpix = cv2.cornerSubPix(gray, np.float32(circles), (5, 5), (-1, -1), criteria)
  corners_subpix = corners_subpix * scaling
  # Check the output format and write the dot center positions to the output file
  if output_format == 'csv':
    # Open the output file in write mode
    corners_subpix = np.around(corners_subpix, decimals=1)
    with open(output_file_name + ".csv", 'w', newline='') as csv_file:
      # Create a CSV writer object
      writer = csv.writer(csv_file)

      # Write the header row
      writer.writerow(['Index', 'X', 'Y'])

      # Iterate over the circles and write the index and center position of each dot to the file
      for i, (x, y) in enumerate(corners_subpix[0,:]):
        writer.writerow([i, x, y])
  elif output_format == 'tsv':
    # Open the output file in write mode
    corners_subpix = np.around(corners_subpix, decimals=1)
    with open(output_file_name + ".tsv", 'w', newline='') as tsv_file:
      # Iterate over the circles and write the index and center position of each dot to the file
      for i, (x, y) in enumerate(corners_subpix[0,:]):
        tsv_file.write(f"{i}\t{x}\t{y}\n")
    
  # Return the positions
  return corners_subpix[:, :]
#-------------------------------------------------------------------------------------------------------------------#
def main(image_file, output_file_name, output_format,scaling):
    if output_format == 'numpy':
        output = get_dot_numpy(image_file,scaling)
    elif output_format == 'tsv' or output_format == 'csv':
        output = get_dot_centers(image_file, output_file_name, output_format, scaling)
    return output

if __name__ == "__main__":
    print(main('test22.png', 'rn_csv', 'tsv', 2))