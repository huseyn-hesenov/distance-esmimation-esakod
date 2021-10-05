import cv2 as cv 
import numpy as np
import time
import math
# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
initialTime=0
initialDistance=0
changeInTime=0
changeInDistance=0
listDistance=[]
listSpeed=[]

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(0,255,0),(0,0,0),(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
GREEN =(0,0,255)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid ==67:
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        # return list 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance
def speedFinder(coveredDistance,timeTaken):
    speed=coveredDistance/timeTaken
    return speed

def averageFinder(completeList,averageOfItems):
    lengthOfList=len(completeList)
    selectedItems=lengthOfList-averageOfItems
    selectedItemsList=completeList[selectedItems :]
    average=sum(range(selectedItems))/len(selectedItemsList)
    return average


# reading the r eference image from dir
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv.namedWindow("izleme sistemi", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("izleme sistemi", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    data = object_detector(frame)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps=int(30)
    fps=str(30)

    for d in data:
        if d[0] =='insan':
            distance = (distance_finder(focal_person, PERSON_WIDTH, d[1]))*2.54
            #listDistance.append(distance)
            #averageDistance=averageFinder(listDistance,6)
            
            distanceInMeters=distance/100
            if initialDistance!=0:
                changeInDistance=initialDistance-distanceInMeters
                if changeInDistance<0:
                    changeInDistance * (-1)
		
                changeInTime=time.time()-initialTime
                speed=speedFinder(coveredDistance=changeInDistance,timeTaken=changeInTime)
                #listSpeed.append(speed)
                #averageSpeed=averageFinder(listSpeed, 10)
                #if(averageSpeed<0):
                    #averageSpeed=averageSpeed* (-1)
                #cv.putText(frame, "obyekt sureti:"+str(averageSpeed), (7, 150), font, 1, (100, 255, 0), 2, cv.LINE_AA)
                speed=speed/1000
                cv.putText(frame, f"Speed: {round(speed, 2)} m/s",(7,170),font ,0.9,(255,122,0),2)
                #print(speed)
                if(speed>1.20):
                    cv.putText(frame,"<<object hereketdedir>>",(7,220),font,0.8,GREEN,1,cv.LINE_AA)

            initialDistance=distance
            initialTime=time.time()
            x, y = d[2]
        elif d[0] =='telefon':
            distance = (distance_finder (focal_mobile, MOBILE_WIDTH, d[1]))*2.54
            x, y = d[2]
        elif d[1]=='stekan':
            distance=(distance_finder(focal_mobile,MOBILE_WIDTH,d[1]))*2.54
            x,y=d[2];
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'obyekt mesafesi: {round(distance,2)} sm', (7,200 ), FONTS, 1, (0,255,0), 2)
        cv.putText(frame, "fps:"+fps, (7, 70), font, 1, (100, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame,"koordinat=>"+ "x:"+str(x)+" "+"y:"+str(y), (7, 110   ), font, 1, (100, 255, 0), 2, cv.LINE_AA)
        #cv.putText(frame, "obyekt sureti:"+averageSpeed, (7, 150), font, 1, (100, 255, 0), 2, cv.LINE_AA)

    t = time.strftime('%H:%M:%S')
    size = cv.getTextSize(t, cv.FONT_HERSHEY_COMPLEX, 1, 2)[0]

    #I1 = cv2.putText(frame.copy(),t,(50,50),font1,0.9,(255,255, 255),3)
    cv.putText(frame, t, (500,size[1]), cv.FONT_HERSHEY_COMPLEX, 0.9, (0,), 9)
    cv.putText(frame, t, (500,size[1]), cv.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255), 1)
    cv.imshow('izleme sistemi',frame)
    #cv.imshow('izleme sistemi',frame)

    #gray = frame

	# resizing the frame size according to our need
    #gray = cv.resize(gray, (500, 300))
	# displaying the frame with fps
   # cv.imshow('frame', gray)


    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

