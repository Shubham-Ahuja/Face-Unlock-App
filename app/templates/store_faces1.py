import cv2
import numpy as np
vid  = cv2.VideoCapture(0)
dataset = cv2.CascadeClassifier("data.xml")
i = 0
face_id_list=open("id_list.txt",'r')
face_id=face_id_list.read()
face_id_list.close()
face_list = []
while True:
    ret, frame = vid.read()
    if ret:
        print(i)
        i+=1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(frame, 1.2)
        for x,y,w,h in faces:
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255,0), 2)
        cv2.imwrite(f"face_{face_id}.png", face)
        face = cv2.resize(face, (50, 50))
        face_list.append(face)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == 27 or i == 50:
            break
    else:
        print("Camera Not Found")
np.save(f"faces/user_{face_id}.npy", np.array(face_list))
next_id=str(int(face_id)+1)
face_id_list=open("id_list.txt",'w')
face_id_list.write(next_id)
face_id_list.close()
vid.release()
cv2.destroyAllWindows()