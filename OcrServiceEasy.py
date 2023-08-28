import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import re
import csv
import matplotlib.pyplot as plt
import os
import io
import numpy as np
import easyocr
import sys
import tkinter.messagebox
import time

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
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        return gray_image
    
    @staticmethod
    def validate_string(input_string):
        allowed_characters = "0123456789.,"
        return all(char in allowed_characters for char in input_string)

    def process_video(self):

        #Open the video file
        video_cap = cv2.VideoCapture(self.file_name)

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

                    processed_image = self.preprocess_frame(frame, self.x_begin, self.width, self.y_begin, self.height)
                    
                    frame_num += 1
                    print(f'Reading frame: {frame_num}')
                    

                    result = reader.readtext(processed_image)

                    if not result:
                        value = np.nan
                    else:
                        value = result[0][1] #extracts the detected value

                        value = value if self.validate_string(value) else np.nan
                    
                    if value == np.nan:
                        self.skipped_frames += 1

                    self.value_per_frame.append(value)

                    self.time_stamps.append(video_cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                    frame_count = 0
                else:
                    frame_count += 1

            else:
                break

        print(f"Skipped Frames: {self.skipped_frames} (No detections or illegal detections)")

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
        base_name = os.path.splitext(self.video_name)[0]  # Get the base name without extension
        plot_file_name = f'plots/{base_name}_plot.png'

        num_frames = [i for i in range(1, len(self.value_per_frame) + 1)]

        plt.plot(num_frames, self.value_per_frame)
        plt.title('Values per Frame')
        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.grid()

        plt.savefig(plot_file_name)

        plt.clf()

        return plot_file_name

    def create_csv(self):
        """
        Save a CSV file to the media folder with two columns: time_stamps, value_per_frame.
        :returns: file path from Django media folder to the csv file
        """
        base_name = os.path.splitext(self.video_name)[0]  # Get the base name without extension
        file_name = f'csvs/{base_name}_data.csv'

        # Create a CSV file in memory
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file, lineterminator = '\n')

        # Write the header
        csv_writer.writerow(['time_stamp_in_sec', 'value_per_frame'])

        # Write the data
        csv_writer.writerows(zip(self.time_stamps, self.value_per_frame))

        with open(file_name, 'w') as file:
            file.write(csv_file.getvalue())

        return file_name
    
class OcrGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HeartReadr OCR GUI")

        self.root.geometry("1280x1000") 

        # Create a frame for the banner
        self.banner_frame = tk.Frame(root, bg="magenta")
        self.banner_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Add the "HeartReadr" label to the banner
        heartreadr_label = tk.Label(self.banner_frame, text="HeartReadr", fg="white", bg="magenta", font=("Helvetica", 24, "bold"))
        heartreadr_label.pack(side="left", padx=10, pady=5)

        icon_image = Image.open("icon/heartreader-icon.png")
        icon_photo = ImageTk.PhotoImage(icon_image)
        text_label_height = heartreadr_label.winfo_reqheight()  # Get the height of the text label
        icon_image = icon_image.resize((text_label_height, text_label_height), Image.ANTIALIAS)  # Resize the icon image
        self.icon_photo = ImageTk.PhotoImage(icon_image)

        # Create a space for an icon on the far right
        self.icon_label = tk.Label(self.banner_frame, image=self.icon_photo, bg="magenta")
        self.icon_label.image = self.icon_photo  # Keep a reference
        self.icon_label.pack(side="right", padx=10, pady=5)

        self.file_button = tk.Button(root, text="Browse MP4 File", command=self.open_file)
        self.file_button.grid(row=1, column=0, pady=10, padx=10)

        self.load_button = tk.Button(root, text="Load Video", command=self.load_video, state=tk.DISABLED)
        self.load_button.grid(row=1, column=1, pady=10, padx=10)

        self.canvas = tk.Canvas(root, width=1280, height=720)
        self.canvas.grid(row=2, column=0, columnspan=2)

        # Create a label to display the instructions
        instructions = (
            "Instructions:\n"
            "1. Press 'Browse MP4 File' and select a file.\n"
            "2. Press 'Load Video' after selecting the file.\n"
            "3. Draw a region from top left to bottom right.\n"
            "4. Press 'Submit OCR Region' and wait for video processing."
        )
        self.instructions_label = tk.Label(root, text=instructions, fg="blue", font=("Helvetica", 12), relief=tk.RAISED, borderwidth=2)
        self.instructions_label.grid(row=3, column=0, columnspan=2, pady=10, padx=10)

        self.submit_button = tk.Button(root, text="Submit OCR Region", command=self.submit_roi, state=tk.DISABLED)
        self.submit_button.grid(row=4, column=0, columnspan=2, pady=10)

    def open_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])
        if self.file_path:
            self.load_button.config(state=tk.NORMAL)

    def load_video(self):
        self.video_cap = cv2.VideoCapture(self.file_path)
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            self.display_frame()
            self.canvas.bind("<ButtonPress-1>", self.start_roi)
            self.canvas.bind("<B1-Motion>", self.draw_roi)
            self.canvas.bind("<ButtonRelease-1>", self.end_roi)
            self.submit_button.config(state=tk.NORMAL)

    def display_frame(self, show_loading=False):
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Calculate the desired width and height for the resized image
        canvas_width = 1280
        canvas_height = 720

        # Calculate the aspect ratio of the original image
        original_aspect_ratio = img.width / img.height

        # Calculate the maximum width and height that fit within the canvas
        max_width = min(img.width, canvas_width)
        max_height = min(img.height, canvas_height)

        # Resize the image while maintaining the aspect ratio
        if original_aspect_ratio > canvas_width / canvas_height:
            resized_img = img.resize((max_width, int(max_width / original_aspect_ratio)), Image.ANTIALIAS)
        else:
            resized_img = img.resize((int(max_height * original_aspect_ratio), max_height), Image.ANTIALIAS)

        self.img_tk = ImageTk.PhotoImage(image=resized_img)

        # Update the canvas size and display the resized image
        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def start_roi(self, event):
        self.x_begin, self.y_begin = event.x, event.y

    def draw_roi(self, event):
        self.canvas.delete("roi_rect")

        x_end, y_end = event.x, event.y

        # Ensure that the x and y coordinates are within canvas bounds
        x_end = max(0, min(x_end, self.canvas.winfo_width()))
        y_end = max(0, min(y_end, self.canvas.winfo_height()))

        if x_end > self.x_begin and y_end > self.y_begin:
            # Calculate the new x_begin, y_begin, width, and height values
            self.width = x_end - self.x_begin
            self.height = y_end - self.y_begin

            # Ensure the ROI box is displayed correctly
            self.canvas.create_rectangle(
                self.x_begin,
                self.y_begin,
                x_end,
                y_end,
                outline="red",
                tags="roi_rect",
            )
        else:
            # Reset the ROI drawing if not starting from top left
            self.width = 0
            self.height = 0

            # Update x_begin and y_begin to the current mouse position
            self.x_begin = x_end
            self.y_begin = y_end


    def end_roi(self, event):
        x_end, y_end = event.x, event.y

        # Ensure that the x and y coordinates are within canvas bounds
        x_end = max(0, min(x_end, self.canvas.winfo_width()))
        y_end = max(0, min(y_end, self.canvas.winfo_height()))

        self.width = x_end - self.x_begin
        self.height = y_end - self.y_begin

        # Calculate the aspect ratio of the original image
        original_aspect_ratio = self.current_frame.shape[1] / self.current_frame.shape[0]

        # Calculate the adjusted coordinates and dimensions for the original image
        adjusted_x_begin = int(self.x_begin / self.canvas.winfo_width() * self.current_frame.shape[1])
        adjusted_width = int(self.width / self.canvas.winfo_width() * self.current_frame.shape[1])
        adjusted_y_begin = int(self.y_begin / self.canvas.winfo_height() * self.current_frame.shape[0])
        adjusted_height = int(self.height / self.canvas.winfo_height() * self.current_frame.shape[0])

        # Update the attributes with adjusted coordinates and dimensions
        self.x_begin = adjusted_x_begin
        self.width = adjusted_width
        self.y_begin = adjusted_y_begin
        self.height = adjusted_height

        self.roi_selected = True


    def submit_roi(self):
        if self.roi_selected:
            self.video_cap.release()
            ocr_service = OcrService(self.file_path, self.x_begin, self.width, self.y_begin, self.height)
            self.process_video_and_finalize(ocr_service)
            
    def process_video_and_finalize(self, ocr_service):
        start_time = time.time()
        ocr_service.process_video()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
        ocr_service.create_csv()
        ocr_service.plot_values()
        tkinter.messagebox.showinfo("Processing Finished", "Video processing is complete!")
        self.root.destroy()
        sys.exit()


def create_directories_if_not_exist():
    directories = ['csvs', 'plots']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

if __name__ == "__main__":
    create_directories_if_not_exist()
    root = tk.Tk()
    app = OcrGUI(root)
    root.mainloop()