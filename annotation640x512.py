# This file is used when the input video size is 640x512 (displayed at size 640x512).
# All information is saved at the normal size of 1280x1024.

import cv2
import numpy as np
import argparse
import os
import subprocess
from tkinter import filedialog
import tkinter as tk

# Construct the argument parser
ap = argparse.ArgumentParser()

# Function to get the video file using a file dialog
def get_video_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4")])
    return file_path

# Get video file using file dialog
video_file = get_video_file()

# Check if a file is selected
if not video_file:
    print("No file selected. Exiting.")
    exit()

cap = cv2.VideoCapture(video_file)

# Create a folder "det" for the detections in the same location as input video:
path_to_detection_folder, _ = os.path.split(video_file)
new_path = os.path.join(path_to_detection_folder, 'gt')

if not os.path.exists(new_path):
    os.mkdir(new_path)

# mouse callback function
global click_list
global positions
positions, click_list = [], []


def callback(event, x, y, flags, param):
    if event == 1:
        click_list.append((x, y))
    positions.append((x, y))

            
cv2.namedWindow('img')
cv2.setMouseCallback('img', callback)
image_number = 0

frame_number = 1
object_id = 1  # cannot be 0 or negative

# read first image
ret, img_p = cap.read()

# get width and height of the original frame
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate resize factor, this will be used to correct the bounding boxes and drawing them on a resized scale
rf_w = w / 640  # original fame width / rescaled width
rf_h = h / 512  # original fame height / rescaled height

img_p = cv2.resize(img_p, (640, 512))

prev_frames = []  # List to store previous frames
bounding_boxes = {}  # Dictionary to store bounding boxes for each frame


# Define list of classes
classes = ['A', 'B', 'C', 'D', 'E']
class_index = 0  # Index to track current class

with open('%s/class.txt' % new_path, 'w') as class_file:
    pass
with open('%s/gt.txt' % (new_path), 'w') as out_file:
    # Define a function to remove the last line from a file
    def remove_last_line(file_path):
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            if lines:
                file.seek(0)
                file.truncate()
                file.writelines(lines[:-1])
                
    def count_boxes(frame_number, bounding_boxes):
        return len(bounding_boxes.get(frame_number, []))
    
    # Define function to save frame info to class.txt
    def save_frame_info(frame_number, object_id, class_label, new_path):
        with open('%s/class.txt' % new_path, 'a') as class_file:
            print('%d,%d,%s' % (frame_number, object_id, class_label), file=class_file)
    
    # Function to draw bounding boxes on image
    def draw_boxes(image, boxes):
        for box in boxes:
            # Divide each coordinate by 2
            box_start = (box[0][0] // 2, box[0][1] // 2)
            box_end = (box[1][0] // 2, box[1][1] // 2)
            cv2.rectangle(image, box_start, box_end, (0, 0, 255), 2)
    
    while (cap.isOpened()):
        img = img_p.copy()

        # Display existing bounding boxes for current frame
        draw_boxes(img, bounding_boxes.get(frame_number, []))

        # Inside your main loop
        if len(click_list) > 0:
            mouse_position = positions[-1]

            a = click_list[-1][0], click_list[-1][1]
            b = mouse_position[0], click_list[-1][1]
            cv2.line(img, a, b, (123, 234, 123), 3)

            a = click_list[-1][0], mouse_position[1]
            b = mouse_position[0], mouse_position[1]
            cv2.line(img, a, b, (123, 234, 123), 3)

            a = mouse_position[0], click_list[-1][1]
            b = mouse_position[0], mouse_position[1]
            cv2.line(img, a, b, (123, 234, 123), 3)

            a = click_list[-1][0], mouse_position[1]
            b = click_list[-1][0], click_list[-1][1]
            cv2.line(img, a, b, (123, 234, 123), 3)

        # If there are four points in the click list, save the image
        if len(click_list) == 2:
            # get the top left and bottom right
            a, b = click_list

            # with open('%s/det.txt'%(new_path),'w') as out_file:
            # MOT 16 det,tx format
            # frame id, -1, xmin, ymin, width, height, confidence, -1, -1, -1
            # as our detections are manual, we will set confidence score as 1
            xmin = min(a[0], b[0]) * rf_w
            ymin = min(a[1], b[1]) * rf_h
            xmax = max(a[0], b[0]) * rf_w
            ymax = max(a[1], b[1]) * rf_h
            width = xmax - xmin
            height = ymax - ymin

            # Save bounding box for the current frame
            bounding_boxes.setdefault(frame_number, []).append(((int(xmin), int(ymin)), (int(xmax), int(ymax))))

            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                frame_number, object_id, xmin, ymin, width, height), file=out_file)
            print(f"{frame_number},{object_id},{xmin},{ymin},{width},{height},1,-1,-1,-1")
            
            save_frame_info(frame_number, object_id, classes[class_index], new_path)

            # reset the click list
            click_list = []

        # show the image and wait
        # Tutorial text
        tuto_text = 'Tutorial:    Next (Frame): N          Back (Frame): B          Remove Box: R          Escape: ESC'
        tuto_text_2 = '    Object ID Increase: F     Object ID Decrease: D     Class Switch: C'
        # Press 'esc' to quit
        # Press 'n' for next frame
        # Press 'b' for previous frame
        # Press 'f' for incrementing object id
        # Press 'd' for decrementing object id
        # Press 'r' to remove the last bounding box
        tuto_text_size, _ = cv2.getTextSize(tuto_text, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
        tuto_text_width, tuto_text_height = tuto_text_size
        tuto_text_size_2, _ = cv2.getTextSize(tuto_text_2, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
        tuto_text_width_2, tuto_text_height_2 = tuto_text_size_2
        cv2.rectangle(img, (640, 480), (0, 640), (0, 0, 0), -1)
        cv2.putText(img, tuto_text, (5, 485 + tuto_text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(img, tuto_text_2, (48, 500 + tuto_text_height_2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Show Frame
        frame_text = f"Frame: {frame_number}"
        text_size, _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_width, text_height = text_size
        
        # Show Object ID
        object_id_text = f"Object ID: {object_id}"
        object_id_text_size, _ = cv2.getTextSize(object_id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        object_id_text_width, object_id_text_height = object_id_text_size
        
        # Show Class
        selected_class_text = f" Class: {classes[class_index]}"
        selected_class_text_size, _ = cv2.getTextSize(selected_class_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        selected_class_text_width, selected_class_text_height = selected_class_text_size
        
        # Show Boxes
        num_boxes = count_boxes(frame_number, bounding_boxes)
        box_count_text = f"Boxes: {num_boxes}"
        box_count_text_size, _ = cv2.getTextSize(box_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_count_text_width, box_count_text_height = box_count_text_size

              

        # Draw a filled rectangle as background for frame number
        cv2.rectangle(img, (0, 5), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 25), (20 + box_count_text_width + 3, 30 + box_count_text_height + 10), (0, 0, 0), -1)
        cv2.rectangle(img, (530, 5), (535 + object_id_text_width + 10, 10 + object_id_text_height + 10), (0, 0, 0), -1)
        cv2.rectangle(img, (530, 25), (535 + selected_class_text_width + 40, 30 + selected_class_text_height + 10), (0, 0, 0), -1)
        
        # Draw frame number text
        cv2.putText(img, frame_text, (10, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, box_count_text, (10, 30 + box_count_text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, object_id_text, (535, 10 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)
        cv2.putText(img, selected_class_text, (528, 30 + selected_class_text_height),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        
            
                       
        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        # escape if 'esc' is pressed
        # 27 is the ascii code for 'esc'
        if k == 27:
            break

        # read next image if 'n' is pressed
        # 110 is the ascii code for 'n'
        if k == 110:
            ret, img = cap.read()
            if not ret:
                break
            if len(prev_frames) >= 10:  # Keep only 10 previous frames
                prev_frames.pop(0)
            prev_frames.append(img_p)  # Save previous frame
            img_p = cv2.resize(img, (640, 512))
            frame_number += 1

        # Move to previous frame if 'b' is pressed
        if k == 98:  # 98 is the ascii code for 'b'
            if prev_frames:
                img_p = prev_frames.pop()  # Retrieve previous frame from the list
                frame_number -= 1
        
              
        # code to increment object id
        # press 'f' to increment object ID
        # 102 is ascii code for 'f'
        if k == 102:
            object_id += 1
            print("object_id incremented to %d" % (object_id))

        # code to decrement object id
        # press 'd' to decrement object ID
        # 100 is ascii code for 'd'
        if k == 100:
            object_id -= 1
            if object_id < 1:
                object_id = 1  # Ensure object_id is not less than 1
            print("object_id decremented to %d" % (object_id))

        
               
        # Cycle through classes when 'c' is pressed 
        if k == 99:  # 99 is ascii code for 'c'
            class_index = (class_index + 1) % len(classes)
            print(f"Class: {classes[class_index]}")

        # Remove last bounding box and last line in gt.txt if 'r' is pressed
        if k == 114:  # 114 is the ascii code for 'r'
            if frame_number in bounding_boxes:
                bounding_boxes[frame_number].pop()  # Remove last bounding box
                remove_last_line('%s/gt.txt' % (new_path))  # Remove last line in gt.txt
                print("Last bounding box removed")
                print("Last bounding box removed", file=out_file)

cap.release()
cv2.destroyAllWindows()
