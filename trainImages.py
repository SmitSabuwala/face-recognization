import face_recognition
import cv2
import os
import config.configHelper as configHelper

def getTrainingImages(helper):
    # define a video capture object
    vid = cv2.VideoCapture(0)
    i = 0
    while (True):

        # Capture the video frame
        # by frame

        ret, frame = vid.read()
        noOfFaceDetections = face_recognition.face_locations(frame)
        if len(noOfFaceDetections) > 0 and i <helper.trainImageCollectionNo:
            print("save the image count:{}".format(i))
            directory = "TrainImages/{}/".format(helper.trainClassName)
            if not os.path.exists(directory):
                os.makedirs(directory)
            print(directory)
            cv2.imwrite("{}/{}.jpg".format(directory,i),frame)
            i += 1
        if i >=helper.trainImageCollectionNo:
            break
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__=="__main__":
    helper = configHelper.get()
    getTrainingImages(helper)