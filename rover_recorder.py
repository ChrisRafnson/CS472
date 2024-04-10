# for realsense frame number info - https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.get_frame_number
# for datetime info - https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
# for how to add aheader to a csv file - https://stackoverflow.com/questions/20347766/pythonically-add-header-to-a-csv-file
# for using random integers - https://www.w3schools.com/python/ref_random_randint.asp
# for finding an element in a list of dictionaries - https://www.geeksforgeeks.org/check-if-element-exists-in-list-in-python/
# for converting to black and white image - https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
# for cropping images - https://learnopencv.com/cropping-an-image-using-opencv/


import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import datetime
import rosbag
import csv
import random
import time

#initialize camera settings and turn on camera
def initialize_pipeline(run):
    pipeline = rs.pipeline()
    config = rs.config()
    #record video to bag file
    config.enable_record_to_file(f'/media/usafa/drone_data/rover_data/{run}.bag')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

#use to connect to rover in future but not used right now
def connect_device(s_connection, b):
    print("Connecting to device...")
    device = connect(s_connection, wait_ready=True, baud = b)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device

#gets color image frame from stream, displays it, and returns the frame number for use in data organization
def get_video_data(pipeline):

    frame = pipeline.wait_for_frames() #get frame
    color_frame = frame.get_color_frame()

    if not color_frame: #if no frame is available signal to restart loop
        return None

    color_image = np.asanyarray(color_frame.get_data()) #get image from frame data
    color_image_fn = color_frame.get_frame_number()
    # cv2.imshow('color', color_image) #display image

    return color_image_fn

#in future collect relavent data form the rover, right now return random ints for each field
def get_rover_data(rover):
    # throttle = random.randint(0, 2000)
    # steering = random.randint(0, 2000)
    # heading = random.randint(0, 359)

    if (not rover.channels['3'] is None
            and not rover.channels['1'] is None):
        throttle = int(rover.channels['3'])
        steering = int(rover.channels['1'])

    heading = rover.heading

    return [throttle, steering, heading]

#add data points to the csv file
def append_ardu_data(throttle, steering, heading, idx, file):
    f = open(file, "a+")
    f.write(f"{idx}, {throttle},{steering},{heading}\n")
    f.close()

def append_data(data, index, data_file):
    field_names = ['index', 'throttle', 'steering', 'heading']
    data_dict = {'index': index, 'throttle': data[0], 'steering': data[1], 'heading': data[2]}
    csv.DictWriter(data_file, fieldnames=field_names).writerow(data_dict)

def main():
    port = "/dev/ttyUSB0"
    baud = 115200
    rover = connect_device(port, baud)
    print("Waiting to Arm")
    while True:
        while not rover.armed:
            time.sleep(3)

        #get starting timestamp for file naming
        print("Arming Drone...")
        run = datetime.datetime.now() 

        #create csv and add headers
        header = ['index', 'throttle', 'steering', 'heading','\n']

        # with open(f'/media/usafa/drone_data/rover_data/{run}.csv', 'w+') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)
        data_file = '/media/usafa/drone_data/rover_data/' + str(run) + '.csv'
        #reopen csv for the remainder of the run
        data_file.write(header)
        pipeline = initialize_pipeline(run)

        while rover.armed:
            print("Rover Armed - Recording Video...")
            #get index of current frame
            index = get_video_data(pipeline)

            #restart loop if no frame was available
            if index == None:
                continue

            #get data from rover: throttle, steering, heading
            data = get_rover_data(rover)
            #add data to csv with current frame index
            append_data(data[0], data[1], data[2], index, data_file)

        print("Recording Finished - Closing File")
        pipeline.stop()
        time.sleep(10)
        #close csv file
        data_file.close()
        print('done')

        #exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


main()