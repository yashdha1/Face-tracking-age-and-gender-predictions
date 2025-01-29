import cv2 
import numpy as np 
from facenet_pytorch import MTCNN

detector = MTCNN()
cam = cv2.VideoCapture(1) 

gender_arci = "modelWeights/deploy_gender.protext" 
gender_weights = "modelWeights/gender_net.caffemodel" 

age_arci = "modelWeights/deploy_age.protext" 
age_weights = "modelWeights/age_net.caffemodel" 

genderDetector  = cv2.dnn.readNetFromCaffe(
    gender_arci,  # Pretriained Model 
    gender_weights  # Pretrained Weights 
)

ageDetector  = cv2.dnn.readNetFromCaffe(
    age_arci,  # Pretriained Model 
    age_weights  # Pretrained Weights 
)

genders = ["male", "female"] 
age = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

while True:
    ret, frame = cam.read()
    if ret:
        boxes, probs = detector.detect(frame)
        if boxes is not None: 
          # Plot insivisual Faces in the FRAME: 
            for box, prob in zip(boxes, probs):
                # Extract box coordinates and convert to integers
                
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                face = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] 
                if face.size == 0:  # Handle cases where the face region is invalid
                    continue
                blob = cv2.dnn.blobFromImage(face, 
                                             scalefactor=1.0, 
                                             size=(227, 227),   
                                             mean=(78.4263377603, 87.7689143744, 114.895847746),   
                                             swapRB=False, 
                                             crop=False)
                
                # making predictions 
                genderDetector.setInput(blob) 
                Gpreds = genderDetector.forward()
                gender = genders[np.argmax(Gpreds)]   
                
                # Age Predictions 
                ageDetector.setInput(blob) 
                Apreds = ageDetector.forward()
                ageFinal = age[np.argmax(Apreds)]   
                
                frame = cv2.rectangle(frame, 
                                      (x1, y1), 
                                      (x2, y2), 
                                      (0, 255, 0), 
                                      5)
                
                # for the face detections 
                frame = cv2.putText(frame, 
                                    text=f"{gender} | {ageFinal}",   
                                    org=(x1, y1 - 10),  
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                    fontScale=1,  
                                    color=(255, 255, 255), 
                                    thickness=2)

        # Show the frame
        cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Exited without any Errors!!!")
        break

# Release camera and destroy all windows
cam.release()
cv2.destroyAllWindows()