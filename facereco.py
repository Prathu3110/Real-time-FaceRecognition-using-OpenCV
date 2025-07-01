import face_recognition
import face_recognition_models
import cv2
import numpy as np
import os

path='faces'

images=[]
classname=[]

# this is just for reading the name image ffile

for img in os.listdir(path):
    image=cv2.imread(f'{path}/{img}')
    images.append(image)
    classname.append(os.path.splitext(img)[0])

print(classname)

#function to make encodings of image from the faces file

def Findencoding(images):
    encodinglist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        print(encode)
        encodinglist.append(encode)
    return encodinglist

encodes=Findencoding(images)
print("Encoding complete...")


# detecting face from webcam

scale=0.25
box_multiplier=1/scale

cap=cv2.VideoCapture(0)

while True:
    success, img= cap.read()

    cur_image=cv2.resize(img,(0,0),fx=scale,fy=scale)
    cur_image=cv2.cvtColor(cur_image,cv2.COLOR_BGR2RGB)

    face_loc= face_recognition.face_locations(cur_image, model='cuda')
    face_encodes=face_recognition.face_encodings(cur_image,face_loc)

# to find matches in faces

    for encodeFaces,faceLocation in zip(face_encodes,face_loc):
        matches= face_recognition.compare_faces(encodes,encodeFaces, tolerance=0.5)
        face_dis=face_recognition.face_distance(encodes,encodeFaces)
        matchindex=np.argmin(face_dis)
        
        if matches[matchindex]:
            name=classname[matchindex].upper()
        else:
            name="unknown"
#draw the box around the face detection
        y1,x2,y2,x1=faceLocation
        y1,x2,y2,x1=int(y1*box_multiplier),int(x2*box_multiplier), int(y2*box_multiplier), int(x1*box_multiplier)
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Webcam Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
