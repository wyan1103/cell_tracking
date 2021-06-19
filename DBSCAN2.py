import numpy as np
import cv2  
import matplotlib.pyplot as plt
import math
import random
import sys

def gen_rand_color():
    intensity1 = random.randint(0, 255)
    intensity2 = random.randint(0, 255)
    color = [intensity1, intensity2, 0]
    random.shuffle(color)
    return color

class Cell:
    def __init__(self, pos, avg_dir, num, frame_num):
        dir_scale = 2
        self.num = num
        self.pos = pos
        self.dir = dir_scale * avg_dir
        self.dir_history = []
        self.pos_history = [(frame_num, pos)]
        self.misses = 0
        self.color = gen_rand_color()
    
    def update_pos(self, new_pos, frame_num):
        self.misses = 0
        self.pos_history.append((frame_num, new_pos))
        self.dir = new_pos - self.pos
        self.dir_history.append(self.dir)
        self.pos = new_pos

    def interp_pos(self, frame_num):
        self.misses += 1
        self.dir = self.avg_dir_history()
        self.pos = self.pos + self.dir
        self.pos_history.append((frame_num, self.pos))
        self.dir_history.append(self.dir)

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

    def __str__(self):
        return f'Cell #{self.num}'

    def __repr__(self):
        return f'Cell #{self.num}'


def dist(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

def vector_mag(A):
    return math.sqrt(A[0] ** 2 + A[1] ** 2)

def vector_angle(A):
    return math.atan2(A[1], A[0])

debug = False
if len(sys.argv) == 1:
    path = "CellTrimFast2.avi"
elif len(sys.argv) == 2:
    path = sys.argv[1]
else:
    path = sys.argv[1]
    debug = True if sys.argv[2] == "-f" else False

video = cv2.VideoCapture(path)
success, frame = video.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_num = 0

cells_per_frame = []
frame_numbers = []

tracked_cells = []
avg_dir = np.asarray([50, 10])

background_frames = [prev_frame]
frame_tracked_cells = []
binary_frames = []

cell_count = 0
cell_total = 0
while(success):
    #print(f'\n=== FRAME {frame_num} ===\n\n')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a difference frame between foreground and an averaged background
    temp_frame = gray.astype(np.int32)
    temp_background = np.average(np.asarray(background_frames), axis=0) #base_frame.astype(np.int32)
    diff_frame = np.clip(temp_frame - temp_background, 0, 255)
    diff_frame = np.abs(diff_frame - np.mean(diff_frame))

    normalized = cv2.normalize(diff_frame.astype(np.uint8), np.zeros_like(diff_frame), 0, 255, cv2.NORM_MINMAX)

    # Create a distance transform to reduce noise and normalize again
    dist = cv2.distanceTransform(diff_frame.astype(np.uint8), cv2.DIST_L2, 3).astype(np.uint8)
    dist = cv2.medianBlur(dist, 5)
    normalized_dist = cv2.normalize(dist, np.zeros_like(dist), 0, 255, cv2.NORM_MINMAX)
    #cv2.imwrite(f'temp/dist{frame_num}.jpg', normalized_dist)

    # Threshold the distance transform and perform morphological operations to clean up.
    # Kernels are elongated under the assumption that faster cells leave longer tails.
    _, binary = cv2.threshold(normalized_dist, 4, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3))
    binary = cv2.dilate(binary, kernel, iterations=3)
    binary = cv2.erode(binary, kernel, iterations=3)
    #binary = cv2.dilate(binary, np.ones((4,4), np.uint8))
    #cv2.imwrite(f'temp/binary{frame_num}.jpg', binary)

    # Find contours in the binary ivector_mage, removing those with small area
    CONTOUR_MIN_AREA = 80
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_contours = []
    contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for c in contours:
        if cv2.contourArea(c) >= CONTOUR_MIN_AREA:
            cell_contours.append(c)

    # Sort by area under the assumption that larger contours are more likely to be cells
    cell_contours = sorted(cell_contours, key=lambda c: cv2.contourArea(c))
    if cell_contours:
        cv2.drawContours(contour_img, cell_contours, -1, (0, 255, 0), 3)
        for c in cell_contours:
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            cv2.circle(contour_img, (cX, cY), 7, (0, 0, 255), 2)
        #cv2.imwrite(f'temp/contours{frame_num}.jpg', contour_img)
    binary_frames.append(contour_img)

    # Generate centroids for each contour
    cell_centroids = []
    for c in cell_contours:
        M = cv2.moments(c)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        cell_centroids.append(np.asarray([cX, cY]))

    ''' TODO: Tracking algorithm is naively implemented and not optimized '''

    MAX_DST_ERROR = 1.5
    MAX_DIR_ERROR = math.pi / 6

    # Update tracked cells based on centroids
    if len(tracked_cells) == 0:
        for cent in cell_centroids:
            tracked_cells.append(Cell(cent, avg_dir, cell_count, frame_num))
            cell_count += 1
            cell_total += 1
    else:
        # Naively check all pairs of cells and centroids, sorting by deviation from expected positions
        pairs = []
        for cell in tracked_cells:
            for cent in cell_centroids:
                predicted_dir = cell.avg_dir_history()
                predicted_pos = cell.pos + predicted_dir
                predicted_angle = vector_angle(predicted_dir)
                actual_angle = vector_angle(cent - cell.pos)

                dst_error = vector_mag(cent - predicted_pos) 
                dir_error = abs(predicted_angle - actual_angle)

                expected_dist = max(vector_mag(predicted_dir), vector_mag(avg_dir))
                if dst_error < expected_dist * MAX_DST_ERROR and dir_error < MAX_DIR_ERROR:
                    pairs.append((cell, cent, dst_error, dir_error))

        pairs = sorted(pairs, key=lambda x: x[2] * x[3])


        # Use the pairs to update each cell with their new positions
        updated_cells = set()
        updated_cents = set()

        for (cell, cent, dst_error, dir_error) in pairs:
            if cell.num not in updated_cells and tuple(cent) not in updated_cents:
                cell.update_pos(cent, frame_num)
                updated_cells.add(cell.num)
                updated_cents.add(tuple(cent))

        # Update the tracked cells if they missed a frame or moved off screen
        MAX_MISSES = 4
        (height, width) = gray.shape

        cells_to_remove = []
        for cell in tracked_cells:
            if cell.num not in updated_cells:
                cell.interp_pos(frame_num)
                (cx, cy) = (cell.pos[0], cell.pos[1])
                if cell.misses > MAX_MISSES:
                    cells_to_remove.append((cell, True))
                elif not (0 <= cx < width and 0 <= cy < height):
                    cells_to_remove.append((cell, False))
                else:
                    updated_cells.add(cell.num)

        for (cell, false_positive) in cells_to_remove:
            tracked_cells.remove(cell)
            if false_positive:
                l = len(frame_tracked_cells)
                for (fnum, prev_pos) in cell.pos_history[:-1]:
                    frame_tracked_cells[fnum].remove(cell)
                cell_total -= 1

        assert(len(updated_cells) == len(tracked_cells))


        # Add the remaining centroids as new tracked cells if they lie within the first 1/4 of the frame (assuming cells flow from left to right)
        for cent in cell_centroids:
            if tuple(cent) not in updated_cents and cent[0] < width // 4:
                updated_cents.add(tuple(cent))
                tracked_cells.append(Cell(cent, avg_dir, cell_count, frame_num))
                cell_count += 1
                cell_total += 1

    frame_tracked_cells.append(tracked_cells.copy())


    if len(tracked_cells) != 0:
        new_avg = np.asarray([0,0], dtype=np.int32)
        for c in tracked_cells:
            new_avg += c.dir
        new_avg //= len(tracked_cells)
        avg_dir = new_avg

    '''
    orig_tracking_img = contour_img.copy()
    for cell in tracked_cells:
        orig_tracking_img = cv2.circle(orig_tracking_img, cell.pos_history[0][1], 6, cell.color, -1)
        for i in range(1, len(cell.pos_history)):
            f1, p1 = cell.pos_history[i-1]
            f2, p2 = cell.pos_history[i]
            orig_tracking_img = cv2.line(orig_tracking_img, p1, p2, cell.color, 3)
            orig_tracking_img = cv2.circle(orig_tracking_img, p1, 6, cell.color, -1)
            orig_tracking_img = cv2.circle(orig_tracking_img, p2, 6, cell.color, -1)
        orig_tracking_img = cv2.putText(orig_tracking_img, str(cell.num), cell.pos, cv2.FONT_HERSHEY_SIMPLEX, 1, cell.color, 2)
    cv2.imwrite(f'temp/tracking_orig{frame_num}.jpg', orig_tracking_img)
    '''

    ''' End tracking algorithm '''

    # Add current frame to window of backround frames
    background_frames.insert(0, gray)
    if len(background_frames) > 10:
        background_frames.pop()

    # Add current frame to graph
    frame_numbers.append(frame_num)

    prev_frame = gray
    success, frame = video.read()
    frame_num += 1

# Generate the cells/frame plot, adding a dummy entry for the 0th position
cells_per_frame = list(map(lambda t: len(t), frame_tracked_cells))
cells_per_frame = cells_per_frame[MAX_MISSES:-MAX_MISSES]
frame_numbers = frame_numbers[MAX_MISSES:-MAX_MISSES]
plt.plot(frame_numbers, cells_per_frame)
plt.savefig('plot.png')

# Generate tracking frames
if debug:
    for (fnum, t) in enumerate(frame_tracked_cells):
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

print(cell_total)