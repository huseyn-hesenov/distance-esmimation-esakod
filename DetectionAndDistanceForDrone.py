import cv2 as cv
import time
# Distance constants
KNOWN_DISTANCE =12.12 #INCHES
DRONE_WIDTH = 4 #INCHES
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
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolo-drone.weights', 'yolo-drone.cfg')

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
        if classid ==0: # drone class id
            print(classid)
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
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
# reading the reference image from dir
ref_drone = cv.imread('ReferenceImages/imageDrone3.jpg')
drone_data = object_detector(ref_drone)
drone_width_in_rf = drone_data[0][1]

print(f"drone width in pixels : {drone_width_in_rf}")
# finding focal length
focal_drone = focal_length_finder(KNOWN_DISTANCE, DRONE_WIDTH, drone_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #find frame center
    (h, w) = frame.shape[:2]  # w:image-width and h:image-height
    cv.circle(frame, (w // 2, h // 2),1, (255, 255, 255),-1)
    cv.namedWindow("izleme sistemi", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("izleme sistemi", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    data = object_detector(frame)
    font = cv.FONT_HERSHEY_SIMPLEX
    fps=str(30)
    for d in data:
        if d[0] =='drone':
            distance = (distance_finder(focal_drone, DRONE_WIDTH, d[1]))*2.54
            x, y = d[2]
            cv.line(frame,(w // 2, h // 2),(x+33,y+34),(255,255,255),1)
        cv.putText(frame, f'obyekt mesafesi: {round(distance,2)} sm', (7,200 ), FONTS, 1, (0,255,0), 2)
        cv.putText(frame, "fps:"+fps, (7, 70), font, 1, (100, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame,"koordinat=>"+ "x:"+str(x)+" "+"y:"+str(y), (7, 110), font, 1, (100, 255, 0), 2, cv.LINE_AA)
    t = time.strftime('%H:%M:%S')
    size = cv.getTextSize(t, cv.FONT_HERSHEY_COMPLEX, 1, 2)[0]
    # put time
    cv.putText(frame, t, (500,size[1]), cv.FONT_HERSHEY_COMPLEX, 0.9, (0,), 9)
    cv.putText(frame, t, (500,size[1]), cv.FONT_HERSHEY_COMPLEX, 0.9, (255,255,255), 1)
    #data
    cv.putText(frame, "Altitude:", (38, 38), cv.FONT_HERSHEY_COMPLEX, 0.9, (0,200,200), 2)
    cv.putText(frame, "Velocity:", (38, 68), cv.FONT_HERSHEY_COMPLEX, 0.9, (0,200,200), 2)
    cv.putText(frame, "Compass:", (38, 98), cv.FONT_HERSHEY_COMPLEX, 0.9, (0,200,200), 2)
    cv.putText(frame, "Armed", (50, 450), cv.FONT_HERSHEY_COMPLEX, 0.9, (0,200,100), 2)
    cv.imshow('izleme sistemi',frame)
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

