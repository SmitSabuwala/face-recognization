import face_recognition
import pickle
import cv2
import os
import config.configHelper as configHelper

def getFaceEncodingsForTrainImage(path):
    face_image_name = []
    face_image_encodings = []
    names = os.listdir(path)
    print(names)
    for name in names:
        dirImageNames = os.listdir("{}/{}".format(path, name))
        for imageName in dirImageNames:
            print(imageName)
            image = face_recognition.load_image_file("{}/{}/{}".format(path, name, imageName))
            noOfFaceDetections = face_recognition.face_locations(image)
            if len(noOfFaceDetections) > 0:
                location = list(noOfFaceDetections[0])
                print(location)
                face_encodings = face_recognition.face_encodings(image, noOfFaceDetections)
                for i in face_encodings:
                    face_image_name.append(name)
                    face_image_encodings.append(i)

                print("noOfFaceDetections in {}:{}".format("{}/{}/{}".format(path, name, imageName),len(noOfFaceDetections)))
    return face_image_encodings, face_image_name

def runModelInferance(helper):
    # define a video capture object
    vid = cv2.VideoCapture(0)
    # i = 0
    # classEncodings,className = getFaceEncodingsForTrainImage("encordingImage")
    # print("classEncodings:{}".format(len(classEncodings)))
    while (True):
        model = pickle.load(open('model/model.pkl', 'rb'))
        ret, frame = vid.read()
        noOfFaceLocation = face_recognition.face_locations(frame)
        # print("noOfLocations:{}".format(len(noOfFaceLocation)))
        for location in noOfFaceLocation:
            face_encording = face_recognition.face_encodings(frame, [location])
            # for encordingIndex in range(0,len(classEncodings)):
            #     print("value of i:{}".format(classEncodings[encordingIndex]))
            #     checkFace = face_recognition.compare_faces([classEncodings[encordingIndex]], face_encording[0])
            #     print("checkFace:{}".format(checkFace))
            #     if checkFace[0]:
            #         frame = cv2.rectangle(frame,(location[1],location[2]),(location[3],location[0]),(255, 0, 0),2)
            #         frame = cv2.putText(frame, str(className[encordingIndex]), (location[3], location[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            #         break
            name = model.predict(face_encording)
            probability = model.predict_proba(face_encording)[:, 1]
            print(probability)
            prob = (probability >= helper.thresholdValue).astype(bool)
            if prob:
                frame = cv2.rectangle(frame,(location[1],location[2]),(location[3],location[0]),(255, 0, 0),2)
                frame = cv2.putText(frame, str(name[0]), (location[3], location[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__=="__main__":
    helper = configHelper.get()
    runModelInferance(helper)