import cv2
from PIL import Image
import re
import csv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import io
import numpy as np
import easyocr

class OcrService:

    def __init__(self, file_name, x_begin, width, y_begin, height):
        '''
        Constructor for OCR Service
        :param file_name: the name of the video file to be processed
        :param x_begin, x_end, y_begin, y_end: Coordinates for cropping the OCR Region
        '''

        #Stripped video name without directories
        self.video_name = os.path.basename(file_name)

        #File path of video including directories
        self.file_name = file_name

        #coordinates for ocr region
        self.x_begin = x_begin
        self.width = width
        self.y_begin = y_begin
        self.height = height

        #number of skipped frames
        self.skipped_frames = 0

        #stores the value for each frame
        self.value_per_frame = []

        #stores the timestamp for each frame
        self.time_stamps = []

        self.fps = -1

    @staticmethod
    def extract_numbers(text):
        '''
        Extracts all the numbers from a string
        :param text: string
        :returns: a list containing numbers
        '''
        return re.findall(r'\d+', text)

    @staticmethod
    def preprocess_frame(frame, x, width, y, height):

        #Crops the iamge
        cropped_image = frame[y:(y + height), x:(x + width)]

        return cropped_image

    def process_video(self):

        #Open the video file
        video_cap = cv2.VideoCapture(self.fs.path(self.file_name))

        if not video_cap.isOpened():
            raise ValueError(f"Unable to open {self.file_name}")

        #initializing ocr
        reader = easyocr.Reader(['en'])

        #counter to indicate which frames to store
        #for now, by default, records every 6th frame per second starting from first
        frame_count = 5

        frame_num = 0

        #Iterating through every frame to process
        while video_cap.isOpened():

            ret, frame = video_cap.read()

            if ret == True:

                #Preprocesses frame and adds name to list
                if frame_count == 5:
                    cropped_image = self.preprocess_frame(frame, self.x_begin, self.width, self.y_begin, self.height)
                    
                    frame_num += 1
                    print(f'Reading frame: {frame_num}')
                    

                    result = reader.readtext(cropped_image)

                    value = result[0][1] #extracts the detected value

                    value = value if value.isdigit() else np.nan

                    self.value_per_frame.append(value)

                    self.time_stamps.append(video_cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    frame_count = 0
                else:
                    frame_count += 1

            else:
                break

        self.fps = video_cap.get(cv2.CAP_PROP_FPS)

        video_cap.release()

        self.time_stamps = [round(value, 2) for value in self.time_stamps]
    
    def average_value(self):

        return np.nanmean(self.value_per_frame)
    
    def min_value(self):

        return min(self.value_per_frame)

    def max_value(self):

        return max(self.value_per_frame)
    
    def plot_values(self):
        '''
        Creates and saves plot to Django Media Folder
        :return: the file path from the media folder in string form
        '''

        num_frames = [i for i in range(1, len(self.value_per_frame) + 1)]

        plot_file_name = f'plots/{self.video_name}_plot.png'

        plt.plot(num_frames, self.value_per_frame)
        plt.title('Values per Frame')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.grid()

        plt.savefig(self.fs.path(plot_file_name))

        plt.clf()

        return plot_file_name

    def create_csv(self):
        """
        Save a CSV file to the media folder with two columns: time_stamps, value_per_frame.
        :returns: file path from Django media folder to the csv file
        """
        file_name = f'csvs/{self.video_name}_data.csv'

        # Create a CSV file in memory
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file, lineterminator = '\n')

        # Write the header
        csv_writer.writerow(['time_stamp_in_sec', 'value_per_frame'])

        # Write the data
        csv_writer.writerows(zip(self.time_stamps, self.value_per_frame))

        abs_path = self.fs.path(file_name)
        with open(abs_path, 'w') as file:
            file.write(csv_file.getvalue())

        return file_name
