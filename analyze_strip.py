import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

PEAK_DISTANCE = 400

def analyze_strip(dir):
    # Load the image
    img = cv.imread(dir)
    if img is None:
        print ('Error opening image')
        print ('Usage: smoothing.py [image_name -- default ../data/lena.jpg] \n')
        return -1

    # cv.namedWindow("Original Image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("Original Image", img)
    # k = cv.waitKey(0)

    # Gaussian Smoothing
    smoothed = cv.GaussianBlur(img, (0, 0), 10)
    
    # cv.namedWindow("Gaussian-Smoothed Image", cv.WINDOW_AUTOSIZE)
    # cv.imshow("Gaussian-Smoothed Image", smoothed)
    # k = cv.waitKey(0)

    # Extract V-channel intensities
    hsv = cv.cvtColor(smoothed, cv.COLOR_RGB2HSV)
    v_channel = -1 * hsv[:,:,2]
    N_ROWS, _ = v_channel.shape

    # Traverse through rows of the image and compute the mean intensity
    mean_intensities = np.empty([N_ROWS])
    for i in range(0, N_ROWS):
        mean_v = np.mean(v_channel[i, :])
        mean_intensities[i] = mean_v

    # plt.plot(mean_intensities)
    # plt.ylabel('Mean V-Channel Intensities')
    # plt.xlabel('Strip Rows')
    # plt.show()

    # Peak Detection
    peaks, _ = find_peaks(mean_intensities, distance=PEAK_DISTANCE)
    if len(peaks) < 2:
        print("Peak Detection Error")
        print(peaks)
        return -1
    
    # plt.plot(mean_intensities)
    # plt.plot(peaks, mean_intensities[peaks], "x")
    # plt.ylabel('Mean V-Channel Intensities')
    # plt.xlabel('Strip Rows')
    # plt.show()
    
    # Calculate the T-C ratio
    t = mean_intensities[peaks[1]]
    c = mean_intensities[peaks[0]]
    ratio = (255 + t) / (255 + c)
    return round(ratio, 3)
