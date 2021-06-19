import cv2
import numpy as np
import math
from progressbar import *
import time
import matplotlib.pyplot as plt

THRESH_CUTOFF = 128
NUM_BG_FRAMES = 10

def unsharp_mask(image, kernel_size=(11, 11), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def normalize_img(im):
    im = im.astype(np.int32)
    minval = np.amin(im)
    maxval = np.amax(im)
    if maxval == 0:
        return np.zeros(im.shape)
    else:
        im = (im - minval) * (255 / (maxval - minval))
        return np.floor(im).astype(np.uint8)

def process_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4)) 
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img 

def process_gradient(gradient):
    gradient = cv2.GaussianBlur(gradient, (41,41), 0)
    gradient_norm = np.zeros_like(gradient)
    cv2.normalize(gradient, gradient_norm, 0, 255, cv2.NORM_MINMAX)
    #kernel = np.asarray([[0,-1,0], [-1,8,-1], [0,-1,0]])
    #gradient = cv2.filter2D(gradient, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    binary = cv2.threshold(gradient, 64, 255, cv2.THRESH_BINARY)[1]

    binary = cv2.dilate(binary, np.ones((8,4), np.uint8), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((12,12), np.uint8))
    return binary

                

def gen_slit_img(path):
    # Get the number of frames and slit height in the video 
    video = cv2.VideoCapture(path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    (success, frame) = video.read()
    slit_height = frame.shape[0]
    slit = frame.shape[1] // 2

    # Create the slit image based on dimensions
    slit_im = np.zeros((num_frames, slit_height), np.uint8)
    base_frame = process_frame(frame)

    # Add slits frame by frame, going down the image
    frame_num = 0
    background_frames = [base_frame]

    frame_cell_counts = []
    frame_numbers = []
    #pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], max_value=num_frames)
    #pbar.start()
    while success:
        frame = process_frame(frame)
        cv2.imwrite(f'temp/zframe{frame_num}.jpg', frame)

        # Create a difference frame and normalize it
        temp_frame = frame.astype(np.int32)
        temp_background = np.average(np.asarray(background_frames), axis=0) #base_frame.astype(np.int32)
        diff_frame = temp_frame - temp_background
        diff_frame = np.abs(diff_frame - np.mean(diff_frame))

        gradient_frame = np.zeros_like(diff_frame)
        gradient_frame = cv2.normalize(diff_frame, gradient_frame, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'temp/gradframe{frame_num}.jpg', gradient_frame)

        # Binarize and process the gradient frame
        binary_frame = process_gradient(gradient_frame)

        # Repair the "tails" left behind by cells in the current frame, replacing them with the background from the previous frame
        base_frame = base_frame * (binary_frame // 255) + frame * (cv2.bitwise_not(binary_frame) // 255)

        # Add to the window of background frames
        background_frames.insert(0, frame)
        if len(background_frames) > NUM_BG_FRAMES:
            background_frames.pop()

        # Ignore the first few frames
        if frame_num > NUM_BG_FRAMES:
            # Add to slit image
            slit_im[frame_num] = binary_frame[:, slit]
            cv2.imwrite(f'temp/bingradframe{frame_num}.jpg', binary_frame)

            # Count number of connected components, assumed to be the number of cells
            binary_frame = np.uint8(binary_frame)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, 4, cv2.CV_32S)
            conn_components = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                (cx, cy) = centroids[i]

                cv2.rectangle(conn_components, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(conn_components, (int(cx), int(cy)), 4, (0, 0, 255), -1)  
            cv2.imwrite(f'temp/concomp{frame_num}.jpg', conn_components)  

            frame_cell_counts.append(num_labels-1)
            frame_numbers.append(frame_num)


        """
        # Use blob detector to count cells
        binary_frame = np.uint8(cv2.bitwise_not(binary_frame))
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = False
        params.minArea = 20
        params.filterByCircularity = True
        params.minCircularity = 0
        params.filterByColor = params.filterByCircularity = params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_frame)

        # Save image for reference/debugging
        im_with_keypoints = cv2.drawKeypoints(binary_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f'temp/kpts{frame_num}.jpg', im_with_keypoints)
        """
        frame_num += 1
        (success, frame) = video.read()
        #pbar.update(frame_num)

    # Duplicate rows (to enlarge blobs) and convert to B/W binary image
    slit_im = slit_im[NUM_BG_FRAMES:]
    slit_im = np.repeat(slit_im, repeats=20, axis=0)
    slit_im = cv2.bitwise_not(slit_im)
    cv2.imwrite('slit.jpg', slit_im)

    # All of the cells should appear as rectangles, so we can count the number of corners
    # to determine the number of cells. This also helps with overlapping regions.
    corners = cv2.goodFeaturesToTrack(slit_im, 0, 0.5, 0)

    # Generate the cells/frame plot
    plt.plot(frame_numbers, frame_cell_counts)
    #plt.show()
    plt.savefig('plot.png')
    return len(corners) // 4

import sys

np.set_printoptions(threshold=sys.maxsize)
l = gen_slit_img('CellTrim.avi')
print(f"\nOutput: {l}")