import numpy as np
import cv2  
import matplotlib.pyplot as plt
import math
import random
import sys
import easygui

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
    MAX_FRAME_MISSES = 4

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
    MAX_DST_ERROR = 1.5
    MAX_DIR_ERROR = math.pi / 6
    CLUMP_MIN_AREA_INCREASE = 1.5

    def __init__(self, initial_avg_dir, frame_shape):
        self.avg_dir = initial_avg_dir
        self.tracked_cells = []
        self.tracked_cells_by_frame = []
        (self.frame_height, self.frame_width) = frame_shape

    def get_pair_error(self, cell, cent):
        predicted_dir = cell.avg_dir_history()
        predicted_pos = cell.pos + predicted_dir
        predicted_angle = Vector.angle(predicted_dir)
        actual_angle = Vector.angle(cent - cell.pos)

        dst_error = Vector.magnitude(cent - predicted_pos) 
        dir_error = abs(predicted_angle - actual_angle)

        return (dst_error, dir_error)


    ''' Get all pairs of cells and centroids with their resulting distance/direction errors, sorted by error '''
    def get_sorted_cell_centroid_errors(self, cell_centroids_areas):
        pairs = []
        for cell in self.tracked_cells:
            for (cent, area) in cell_centroids_areas:
                (dst_error, dir_error) = self.get_pair_error(cell, cent)

                predicted_dir = cell.avg_dir_history()
                #predicted_pos = cell.pos + predicted_dir
                #predicted_angle = Vector.angle(predicted_dir)
                #actual_angle = Vector.angle(cent - cell.pos)

                expected_dist = np.mean([Vector.magnitude([predicted_dir]), Vector.magnitude(self.avg_dir)])

                max_dst_error = CellTracker.MAX_DST_ERROR 
                max_dir_error = CellTracker.MAX_DIR_ERROR

                major_axis = expected_dist * 3
                minor_axis = expected_dist * 1

                # If a cell is relatively new, relax the error bounds to account for abnormally fast cells
                if len(cell.dir_history) < CellTracker.NEW_CELL_FRAME_TOLERANCE:
                    max_dst_error *= 2

                if dst_error < expected_dist * max_dst_error and dir_error < max_dir_error:
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
                    cell.update_pos(cent, np.mean([cell.size, area]), frame_num)
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
                        self.tracked_cells_by_frame[fnum].remove(cell)

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
                self.tracked_cells.append(TrackedCell(cent, self.avg_dir, area, frame_num))

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
                self.tracked_cells.append(TrackedCell(cent, self.avg_dir, area, frame_num))

        # Otherwise interpret tracked cells new positions and add new ones if necessary
        else:
            (updated_cells, updated_cents) = self.update_tracked_cells(cell_centroids_areas, frame_num)
            self.update_missing_cells(updated_cells, frame_num)
            self.add_new_cells(cell_centroids_areas, updated_cents, frame_num)

        # Finally, update the tracker history and average cell displacement
        self.tracked_cells_by_frame.append(self.tracked_cells.copy())
        if len(self.tracked_cells) != 0:
            new_avg = np.asarray([0,0], dtype=np.int32)
            for cell in self.tracked_cells:
                new_avg += cell.dir
            self.avg_dir = new_avg // len(self.tracked_cells)


''' 
Processes frames from a video to extract centroid locations of potential cells
'''
class FrameProcessor:
    NUM_BG_FRAMES = 10
    CONTOUR_MIN_AREA = 80

    @staticmethod
    def grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        success, frame = self.video.read()
        if not success:
            print("File is invalid")
            exit(1)

        gray = FrameProcessor.grayscale(frame)
        self.background_frames = [gray]
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
    '''
    General Idea: Given that cells are illuminated from one direction, a cell will always have a 'light' and 'dark'
                  side. We can look at the differences between the current frame and previous frames in order to 
                  identify the 'light' sides (here this is done using np.clip and np.abs) to track cells. 
                  We can then dilate and erode with an elongated kernel since fast moving cells leave behind 'tails' 
                  when looking at difference frames, leading to elongated cell shapes.
    '''
    def get_binary(self, frame):
        gray = FrameProcessor.grayscale(frame).astype(np.int32)

        # Look at the difference between the frame foreground and an averaged background
        avg_background = np.average(np.asarray(self.background_frames), axis=0)
        diff_frame = np.clip(gray - avg_background, 0, 255)
        diff_frame = np.abs(diff_frame - np.mean(diff_frame))

        # Distance transform to reduce noise, normalize to 0-255 
        dist = cv2.distanceTransform(diff_frame.astype(np.uint8), cv2.DIST_L2, 3).astype(np.uint8)
        dist = cv2.medianBlur(dist, 5)
        normalized_dist = cv2.normalize(dist, np.zeros_like(dist), 0, 255, cv2.NORM_MINMAX)

        # Finally threshold and perform morphological operations to clean up holes and noise
        _, binary = cv2.threshold(normalized_dist, 4, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3))
        binary = cv2.dilate(binary, kernel, iterations=3)
        binary = cv2.erode(binary, kernel, iterations=3)

        # Add current frame to a sliding window of background frames
        self.background_frames.insert(0, gray)
        if len(self.background_frames) > FrameProcessor.NUM_BG_FRAMES:
            self.background_frames.pop()

        self.last_binary = binary
        return binary

    ''' Extract blob contours from a binary frame '''
    def get_contours(self, binary):
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_contours = filter(lambda c: cv2.contourArea(c) >= FrameProcessor.CONTOUR_MIN_AREA, contours)
        cell_contours = sorted(cell_contours, key = lambda c: cv2.contourArea(c))

        if DEBUG and cell_contours:
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
            cv2.imwrite(f'temp/zframe{self.frame_num}.jpg', frame)
            binary = self.get_binary(frame)
            contours = self.get_contours(binary)
            self.last_contour = contours
            return True, contours

DEBUG = True
DEBUG_PRINT = False
path = easygui.fileopenbox()
print(path)

if DEBUG:
    binary_frames = []

frame_processor = FrameProcessor(path)

initial_avg_dir = np.asarray([50, 10])
cell_tracker = CellTracker(initial_avg_dir, frame_processor.get_frame_shape())

while(True):
    success, cell_contours = frame_processor.get_next_frame_contours()
    printd(f"\n===== FRAME {frame_processor.frame_num} =====\n")
    if not success:
        break
    else:
        cell_tracker.update_tracker(cell_contours, frame_processor.frame_num)

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



# Generate the cells/frame plot, skipping the first/last few frames
mm = TrackedCell.MAX_FRAME_MISSES
frame_numbers = list(range(frame_processor.frame_num))[mm:-mm]
cells_per_frame = list(map(lambda t: len(t), cell_tracker.tracked_cells_by_frame))[mm:-mm]
plt.plot(frame_numbers, cells_per_frame)
plt.savefig('plot.png')

# Generate tracking frames
if DEBUG:
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

print(TrackedCell.cell_total)
    