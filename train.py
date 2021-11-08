import face_recognition
import cv2
import os
from sklearn import svm
import pickle
import sys
import config.configHelper as configHelper



def getFaceEncodingsForTrainImage(path):
    face_image_name = []
    face_image_encodings = []
    names = os.listdir(train_image_path)
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

if __name__=='__main__':
    helper = configHelper.get()
    train_image_path = helper.trainImagePath
    # train_image_path = str(sys.argv[1])
    face_image_encodings,face_image_name=getFaceEncodingsForTrainImage(train_image_path)
    clf = svm.SVC(gamma ='scale',probability=True)
    clf.fit(face_image_encodings,face_image_name)
    # Pkl_Filename = "Pickle_RL_Model.pkl"
    # with open(Pkl_Filename, 'wb') as file:
    #     pickle.dump(clf, file)
    pickle.dump(clf, open('model/model.pkl', 'wb'))
    # saved_model = pickle.dumps(clf)
