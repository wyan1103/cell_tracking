import numpy as np
import cv2  
import matplotlib.pyplot as plt
import math
import random
import sys
import easygui
import progressbar
import os

# Print if debug flag is set
def printd(text):
    if DEBUG_PRINT:
        print(text)

'''
Vector helper methods, where vectors are implemented as numpy arrays
'''
class Vector:
    @staticmethod
    def dist(A, B):
        return Vector.magnitude(A - B) 

    @staticmethod
    def magnitude(A):
        return np.linalg.norm(A) 

    @staticmethod
    def angle(A):
        return math.atan2(A[1], A[0])


'''
Represents a single tracked cell and its history, with functions to update the cell's position
and direction based on new frames
Note: Cell centers are converted to np arrays for easier calculations.
'''
class TrackedCell:
    MAX_FRAME_MISSES = 5

    cell_total = 0
    cell_nums = set()
    tracking_num = 0

    @staticmethod
    def gen_rand_cell_color():
        R = random.randint(128, 255)
        G = random.randint(128, 255)
        B = random.randint(128, 255)
        return (R,G,B)

    def __init__(self, pos, init_dir, init_size, frame_num):
        self.pos = np.asarray(pos)
        self.dir = init_dir
        self.size = init_size

        self.pos_history = [(frame_num, pos)]
        self.dir_history = []
        self.size_history = []

        self.misses = 0
        self.color = TrackedCell.gen_rand_cell_color()
        
        self.num = TrackedCell.tracking_num
        TrackedCell.tracking_num += 1
        TrackedCell.cell_total += 1
        TrackedCell.cell_nums.add(self.num)

    def update_pos(self, new_pos, new_size, frame_num):
        new_pos = np.asarray(new_pos)
        self.dir = new_pos - self.pos
        self.pos = new_pos
        self.size = new_size

        self.dir_history.append(self.dir)
        self.size_history.append(self.size)
        self.pos_history.append((frame_num, new_pos))

        self.misses = 0

    def interp_pos(self, frame_num):
        self.dir = self.avg_dir_history()
        self.pos = self.pos + self.dir

        self.pos_history.append((frame_num, self.pos))

        self.misses += 1

        if self.misses > TrackedCell.MAX_FRAME_MISSES:
            TrackedCell.cell_total -= 1
            TrackedCell.cell_nums.remove(self.num)
            return True

        return False

    def avg_dir_history(self):
        past_records = 5
        if len(self.dir_history) == 0:
            return self.dir
        elif len(self.dir_history) < past_records:
            return np.mean(self.dir_history, axis=0).astype(np.int32)
        else:
            avg = np.asarray([0,0])
            for i in range(1, past_records + 1):
                avg += self.dir_history[-i]
            avg //= past_records
            return avg.astype(np.int32)

    def avg_size_history(self):
        past_records = 5
        if len(self.size_history) == 0:
            return self.size
        elif len(self.size_history) < past_records:
            return np.mean(self.size_history, axis=0).astype(np.int32)
        else:
            avg = 0
            for i in range(1, past_records + 1):
                avg += self.size_history[-i]
            avg //= past_records
            return avg

    def __str__(self):
        return f'TrackedCell #{self.num}'

    def __repr__(self):
        return f'#{self.num}'


'''
Keeps track of all cells given new frame centroids and frame numbers
Note: Cell centers are represented as tuples to make them hashable.
'''
class CellTracker:
    NEW_CELL_FRAME_TOLERANCE = 4
    CLUMP_MIN_AREA_INCREASE = 1.5

    def __init__(self, initial_avg_dir, frame_shape):
        self.avg_dir = initial_avg_dir
        self.tracked_cells = set()
        self.tracked_cells_by_frame = []
        self.cells_per_frame = []
        (self.frame_height, self.frame_width) = frame_shape

    ''' Get all pairs of cells and centroids with their resulting distance/direction errors, sorted by error '''
    def get_sorted_cell_centroid_errors(self, cell_centroids_areas):
        pairs = []
        for cell in self.tracked_cells:
            # Predict cell movement based on its direction history and the overall average displacement vector
            predicted_dir = cell.avg_dir_history()
            predicted_pos = cell.pos + predicted_dir
            predicted_angle = Vector.angle(predicted_dir)
            expected_dist = Vector.magnitude(predicted_dir) #np.mean([Vector.magnitude([predicted_dir]), Vector.magnitude(self.avg_dir)])

            # We accept a centroid-cell matching if it lies in an ellipse defined by the following parameters
            major_axis = expected_dist * 1.5
            minor_axis = expected_dist * 0.5
            if len(cell.dir_history) < CellTracker.NEW_CELL_FRAME_TOLERANCE:   # more tolerance for new cells
                major_axis *= 2
            (cx, cy) = tuple(predicted_pos)
            ang = -1 * predicted_angle

            # Helper function that checks if a point lies in the search ellipse
            def point_in_search_ellipse(x, y):
                c1 = ((x - cx) * math.cos(ang) - (y - cy) * math.sin(ang)) ** 2 / (major_axis ** 2)
                c2 = ((x - cx) * math.sin(ang) + (y - cy) * math.cos(ang)) ** 2 / (minor_axis ** 2)
                return c1 + c2 <= 1

            # Try to match each centroid to the current cell, calculating error margins and rejecting if it lies beyond the ellipse
            for (cent, area) in cell_centroids_areas:
                actual_angle = Vector.angle(cent - cell.pos)
                dst_error = Vector.magnitude(cent - predicted_pos) 
                dir_error = abs(predicted_angle - actual_angle)

                if point_in_search_ellipse(cent[0], cent[1]):
                    pairs.append((cell, cent, area, dst_error, dir_error))
        
        return sorted(pairs, key = lambda x: x[3] * x[4])


    ''' Match cells with centroids and update their position, returnning sets of matched cells and centroids '''
    def update_tracked_cells(self, cell_centroids_areas, frame_num):
        # Get pairs of cells and centroids and create sets to keep track of cells and centroids we have matched.
        cell_centroid_pair_errors = self.get_sorted_cell_centroid_errors(cell_centroids_areas)
        updated_cells = set()
        updated_cents = set()

        potential_clumps = set()
        clump_areas = dict()

        for (cell, cent, cont_area, dst_error, dir_error) in cell_centroid_pair_errors:
            if cell.num not in updated_cells:
                if cent not in updated_cents:
                    printd(f"Updating #{cell.num}")
                    area = cont_area
                elif cent in potential_clumps:
                    printd(f"Updating Occluded #{cell.num}")
                    area = clump_areas[cent]
                    potential_clumps.remove(cent)
                else:
                    continue

                cell_size = cell.avg_size_history()
                if cell_size * CellTracker.CLUMP_MIN_AREA_INCREASE < area:
                    potential_clumps.add(cent)
                    clump_areas[cent] = area - cell_size
                    cell.update_pos(cent, np.mean([cell_size, area]), frame_num)
                else:
                    cell.update_pos(cent, area, frame_num)

                updated_cells.add(cell.num)
                updated_cents.add(cent)

        return (updated_cells, updated_cents)
        

    ''' Track all unmatched cells by predicting their new location, untracking them if necessary '''
    def update_missing_cells(self, cells_tracked, frame_num):
        cells_to_remove = []
        for cell in self.tracked_cells:
            if cell.num not in cells_tracked:
                cell_missing = cell.interp_pos(frame_num)
                (cx, cy) = (cell.pos[0], cell.pos[1])

                # Stop tracking missing cells and cleanse them from tracker history
                if cell_missing:
                    cells_to_remove.append(cell)
                    printd(f"Removing: #{cell.num}")
                    for (fnum, prev_pos) in cell.pos_history[:-1]:
                        self.cells_per_frame[fnum] -= 1
                     #   self.tracked_cells_by_frame[fnum].remove(cell)

                # Stop tracking cells that have moved offscreen 
                elif not (0 <= cx < self.frame_width and 0 <= cy < self.frame_height):
                    printd(f"Cell Finished: #{cell.num}")
                    cells_to_remove.append(cell)
                else:
                    printd(f"Interpreting: #{cell.num}")

        # Remove the cells in a separate loop to avoid messing up the original
        for cell in cells_to_remove:
            self.tracked_cells.remove(cell)


    ''' Add unmatched centroids as new tracked cells if they are in the first 1/4 of the frame '''
    def add_new_cells(self, cell_centroids_areas, centroids_matched, frame_num):
        for (cent, area) in cell_centroids_areas:
            if cent not in centroids_matched and \
               cent[0] < self.frame_width // 4:
                self.tracked_cells.add(TrackedCell(cent, self.avg_dir, area, frame_num))

    ''' Get contour centroids from a list of contours '''
    def get_centroids_with_areas(self, contours):
        cell_centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            centroid = (cX, cY)
            cell_centroids.append((centroid, area))

        return cell_centroids

    ''' Update tracked cells over the next frame using the given centroids '''
    def update_tracker(self, cell_contours, frame_num):
        cell_centroids_areas = self.get_centroids_with_areas(cell_contours)

        # If we are not tracking any cells, consider all centroids as cells
        if not self.tracked_cells:
            for (cent, area) in cell_centroids_areas:
                self.tracked_cells.add(TrackedCell(cent, self.avg_dir, area, frame_num))

        # Otherwise interpret tracked cells new positions and add new ones if necessary
        else:
            (updated_cells, updated_cents) = self.update_tracked_cells(cell_centroids_areas, frame_num)
            self.update_missing_cells(updated_cells, frame_num)
            self.add_new_cells(cell_centroids_areas, updated_cents, frame_num)

        # Finally, update the tracker history and average cell displacement
        self.cells_per_frame.append(len(self.tracked_cells))
        #self.tracked_cells_by_frame.append(self.tracked_cells.copy())
        if len(self.tracked_cells) != 0:
            new_avg = np.asarray([0,0], dtype=np.int32)
            for cell in self.tracked_cells:
                new_avg += cell.dir
            self.avg_dir = new_avg // len(self.tracked_cells)


''' 
Processes frames from a video to extract centroid locations of potential cells
'''

class FrameProcessor:
    NUM_BG_FRAMES = 50
    CONTOUR_MIN_AREA = 80

    @staticmethod
    def grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def __init__(self, video_path):
        self.totDif = self.totDst = self.totTsh = self.totRem = 0
        self.video = cv2.VideoCapture(video_path)
        success, frame = self.video.read()
        if not success:
            print("File is invalid")
            exit(1)

        gray = FrameProcessor.grayscale(frame)
        self.background_frames = [gray]
        self.background_avg = gray.astype(np.float64)
        self.frame_num = -1  # frames indexed from 0, first is ignored

        self.last_binary = None
        self.last_contour = None

    ''' Return the shape of a frame '''
    def get_frame_shape(self):
        return self.background_frames[0].shape

    ''' Return the last frame '''
    def get_last_contour(self):
        contours = self.last_contour
        frame = cv2.cvtColor(self.last_binary, cv2.COLOR_GRAY2BGR)
        return cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    ''' Process the provided frame into a binary image with cells as white blobs '''
    def get_binary(self, frame):
        gray = FrameProcessor.grayscale(frame)
        gray = cv2.GaussianBlur(gray, (101,101), cv2.BORDER_DEFAULT)
        cv2.imwrite(f'temp/gray{self.frame_num}.jpg', gray)
        #cv2.imshow('hi', gray)
        #cv2.waitKey(0)

        t1 = time.time()

        tmp = np.asarray(self.background_frames)

        t2 = time.time()

        # Look at the difference between the frame foreground and an averaged background
        avg_background = np.average(tmp, axis=0).astype(np.uint8)

        absd = cv2.absdiff(avg_background, gray)
        norm_absd = cv2.normalize(absd, np.zeros_like(absd), 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'temp/abdiff{self.frame_num}.jpg', norm_absd)

        t3 = time.time()

        diff_frame = np.clip(gray - avg_background, 0, 255)

        #diff_frame = np.clip(d, 0, 255, out=growth)

        t4 = time.time()

        diff_frame = np.abs(diff_frame - np.mean(diff_frame))


        # Distance transform to reduce noise, normalize to 0-255 
        dist = cv2.distanceTransform(diff_frame.astype(np.uint8), cv2.DIST_L2, 3).astype(np.uint8)
        dist = cv2.medianBlur(dist, 5)
        normalized_dist = cv2.normalize(dist, np.zeros_like(dist), 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite(f'temp/dist{self.frame_num}.jpg', normalized_dist)

        #cv2.imshow('hi', normalized_dist)
        #cv2.waitKey(0)

        # Finally threshold and perform morphological operations to clean up holes and noise
        _, binary = cv2.threshold(normalized_dist, 16, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3))
        binary = cv2.dilate(binary, kernel, iterations=3)
        binary = cv2.erode(binary, kernel, iterations=3)

        t5 = time.time()

        self.totDif += t2 - t1
        self.totDst += t3 - t2
        self.totTsh += t4 - t3
        self.totRem += t5 - t4

        # Add current frame to a sliding window of background frames
        
        self.background_frames.insert(0, gray)
        if len(self.background_frames) > FrameProcessor.NUM_BG_FRAMES:
            last = self.background_frames.pop()

        self.last_binary = binary
        return binary

    ''' Extract blob contours from a binary frame '''
    def get_contours(self, binary):
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_contours = filter(lambda c: cv2.contourArea(c) >= FrameProcessor.CONTOUR_MIN_AREA, contours)
        cell_contours = list(cell_contours)\

        if DEBUG:
            contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, cell_contours, -1, (0, 255, 0), 3)
            for c in cell_contours:
                M = cv2.moments(c)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                cv2.circle(contour_img, (cX, cY), 7, (0, 0, 255), 2)
            binary_frames.append(contour_img)

        return cell_contours

    ''' Get cell contours from the next frame in the video '''
    def get_next_frame_contours(self):
        success, frame = self.video.read()
        self.frame_num += 1
        if not success: 
            return False, None
        else:
            binary = self.get_binary(frame)
            contours = self.get_contours(binary)
            self.last_contour = contours
            return True, contours

import time

DEBUG = True
DEBUG_PRINT = False
path = easygui.fileopenbox()
print(path)

start_time = time.time()

if DEBUG:
    binary_frames = []

frame_processor = FrameProcessor(path)
length = int(frame_processor.video.get(cv2.CAP_PROP_FRAME_COUNT))

initial_avg_dir = np.asarray([50, 10])
cell_tracker = CellTracker(initial_avg_dir, frame_processor.get_frame_shape())

widgets = [progressbar.AnimatedMarker(), progressbar.Percentage(), progressbar.Bar(), progressbar.AdaptiveETA()]

totTimeFP = 0
totTimeTR = 0


bar = progressbar.ProgressBar(max_value=length, widgets=widgets).start()
while(True):
    startTime = time.time()
    success, cell_contours = frame_processor.get_next_frame_contours()
    totTimeFP += time.time() - startTime
    printd(f"\n===== FRAME {frame_processor.frame_num} =====\n")
    if not success:
        break
    else:
        startTime = time.time()
        cell_tracker.update_tracker(cell_contours, frame_processor.frame_num)
        totTimeTR += time.time() - startTime

        # Output naive tracking frames (tracked cells before false positives are purged)
        if DEBUG:
            contour_frame = frame_processor.get_last_contour()
            for cell in cell_tracker.tracked_cells:
                p1 = p2 = None
                for i in range(len(cell.pos_history) - 1):
                    f1, p1 = cell.pos_history[i]
                    f2, p2 = cell.pos_history[i+1]
                    contour_frame = cv2.circle(contour_frame, tuple(p1), 5, cell.color, -1)
                    contour_frame = cv2.line(contour_frame, p1, p2, cell.color, 3)
                    
                if p2 is None:
                    f1, p1 = cell.pos_history[0]
                    contour_frame = cv2.circle(contour_frame, tuple(p1), 5, cell.color, -1)
                    contour_frame = cv2.putText(contour_frame, str(cell.num), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 1, cell.color, 2)
                else:
                    contour_frame = cv2.circle(contour_frame, tuple(p2), 5, cell.color, -1)
                    contour_frame = cv2.putText(contour_frame, str(cell.num), tuple(p2), cv2.FONT_HERSHEY_SIMPLEX, 1, cell.color, 2)
            
            cv2.imwrite(f'temp/initial_tracking{frame_processor.frame_num}.jpg', contour_frame)
    bar.update(frame_processor.frame_num)

# Create output directory if one doesn't exist
if not os.path.isdir("output/"):
    os.mkdir("output/")

startTime = time.time()

# Generate the cells/frame plot, skipping the first/last few frames
mm = TrackedCell.MAX_FRAME_MISSES
frame_numbers = list(range(frame_processor.frame_num))[mm:-mm]
cells_per_frame = cell_tracker.cells_per_frame[mm:-mm] #list(map(lambda t: len(t), cell_tracker.tracked_cells_by_frame))[mm:-mm]
plt.plot(frame_numbers, cells_per_frame)
plt.savefig('output/plot.png')

# Generate tracking frames
if DEBUG:
    print(len(cell_tracker.tracked_cells_by_frame))
    for (fnum, t) in enumerate(cell_tracker.tracked_cells_by_frame):
        tracking_img = binary_frames[fnum]
        for cell in t:
            for i in range(1, len(cell.pos_history)):
                (f1, p1) = cell.pos_history[i-1]
                (f2, p2) = cell.pos_history[i]

                tracking_img = cv2.circle(tracking_img, p1, 6, cell.color, -1)
                if f1 == fnum:
                    tracking_img = cv2.putText(tracking_img, str(cell.num), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, cell.color, 2)
                    break
            
                tracking_img = cv2.line(tracking_img, p1, p2, cell.color, 3)
                tracking_img = cv2.circle(tracking_img, p2, 6, cell.color, -1)
            

        cv2.imwrite(f'temp/tracking{fnum}.jpg', tracking_img)

end_time = time.time()
time_taken = end_time - start_time

count = TrackedCell.cell_total
fully_tracked = TrackedCell.cell_total - len(cell_tracker.tracked_cells)
still_tracked = count - fully_tracked
dropped_cells = TrackedCell.tracking_num - TrackedCell.cell_total

f = open("output/data.txt", "w")
f.write(path + "\n\n")
f.write(f"Cell Count: {count}\n\n")
f.write(f"Cells Fully Tracked: {fully_tracked}\n")
f.write(f"Cells Being Tracked: {still_tracked}\n")
f.write(f"Cells Dropped: {dropped_cells}\n\n")
f.write(f"Time Taken: {time_taken} seconds")
f.close()

totTimeWR = time.time() - startTime
print("WRITE TIME: " + str(totTimeWR))
print("FRAME TIME: " + str(totTimeFP))
print("TRACKING TIME: " + str(totTimeTR))

print(frame_processor.totDif)
print(frame_processor.totDst)
print(frame_processor.totTsh)
print(frame_processor.totRem)