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

    ''' Process the provided frame into a binary image with cells as white blobs '''
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
        return list(cell_contours)

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