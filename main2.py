import cv2
import pandas as pd
from ultralytics import YOLO
from test import*
import time
import os
import matplotlib.pyplot as plt
from Number_plate_main import extract_number

model=YOLO('yolov8n.pt')
class_list = ['bicycle', 'car', 'motorcycle', 'bus','truck']

tracker = Tracker()
count=0

file_name="video1.mp4"
cap=cv2.VideoCapture(file_name)
frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps= cap.get(cv2.CAP_PROP_FPS)
speed_limit=80
df_over=pd.DataFrame(columns=["Vehicle No","Speed"])
print(frame_width,frame_height)
df=pd.DataFrame(columns=["SNO","Vehicle No","Speed"])
down = {}
down_v={}
up = {}
counter_down = []
counter_up = []
ovs=0


red_line_y = 150
blue_line_y = 500
offset = 4
n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Result.mp4', fourcc, 25.0, (int(frame_width), int(frame_height)))
veh=0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count+=1
    print(count,frame.shape)
    results = model.predict(frame,imgsz=1920)
    print(f"Frame shape after YOLO: {results[0].orig_shape}")

    # print(results)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = results[0].names[int(d)]
        if c in class_list:
            list.append([x1,y1,x2,y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id = bbox
        cx = int(x3+x4)//2
        cy = int(y3+y4)//2
    # print(cx,cy,red_line_y,blue_line_y)
        # cv2.circle(frame,(cx,cy),offset,(0,0,255),-1)

        if red_line_y < (cy+offset) and blue_line_y > (cy-offset) and id not in down:
            down[id]=count
            down_v[id]=frame[y3:y4,x3:x4]
            print(down,id,count)    
        if id in down:
            if blue_line_y < (cy+offset) and blue_line_y > (cy-offset) :
                frames_differ=count-down[id]
                elapsed_time =frames_differ/fps 
                print(elapsed_time,frames_differ,down[id],id)
                down_v[id]=frame[y3:y4,x3:x4]
                if counter_down.count(id)==0:
                    counter_down.append(id)
                    distance = 22.5
                    a_speed_ms = distance/elapsed_time
                    a_speed_kh = a_speed_ms*3.6
                    print(a_speed_kh,"Kmph",id)
                    number_plate=extract_number(down_v[id])
                    vehicle_name=f"detected_frames/Vehicles/vehicle_{number_plate[0]}_{a_speed_kh}.jpg"
                    cv2.imwrite(vehicle_name,down_v[id])
                    veh+=1
                    data={"SNO": veh ,"Vehicle No":number_plate[0],"Speed":a_speed_kh}
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
                    if a_speed_kh > speed_limit:
                        ovs+=1
                        data={"Vehicle No":number_plate[0],"Speed":a_speed_kh}
                        df_over=pd.concat([df,pd.DataFrame([data])],ignore_index=True)
                        
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    print(int(frame_width))
    cv2.line(frame, (0, red_line_y), (int(frame_width), red_line_y), red_color, 2)
    cv2.putText(frame, ('Red Line'), (172, red_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (0, blue_line_y), (int(frame_width), blue_line_y), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (8, blue_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, ('Vehicles - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('overspeedVehicles - ' + str(ovs)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save frame
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)
    out.write(frame)
    cv2.imshow("output",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
df.to_excel("output.xlsx", index=True)
df.to_excel("Overspeed_vehicles.xlsx",index=True)
print("data saved sucessfully")
cap.release()
out.release()
cv2.destroyAllWindows()
