from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import matplotlib.pyplot as plt
from util import get_car, read_plate, write_csv


# pytesseract.pytesseract.tesseract_cmd="C:/Program Files/Tesseract-OCR/tesseract.exe"

# license_plate_detector = YOLO('license_plate_detector.pt')

# frame= cv2.imread(r"C:\Users\RAJESH\Downloads\MINI PROJECT\PROJECT\detected_frames\Vehicles\vehicle_0_28.125.jpg")
def extract_number(frame):
    license_plate_detector = YOLO('license_plate_detector.pt')
    license_plates=license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id = license_plate
        print(x1,x2,y1,y2,score,class_id)


    license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2)]
    # plt.imshow(cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2RGB))
    # mask = np.zeros_like(frame)
    # mask[int(y1):int(y2),int(x1):int(x2)] = frame[int(y1):int(y2),int(x1):int(x2)]

    print(read_plate(license_plate_crop))
    # plt.imshow(cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2RGB))


    # number = read_plate(mask)
    number = read_plate(license_plate_crop)
    return number
    # plt.imshow(cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2RGB))