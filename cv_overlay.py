import os
import cv2
import numpy as np

projectroot = os.path.expanduser("~/SciProjects/Project_FIPER/")
speedo = 255 - cv2.imread(projectroot + "speedometer.jpg")  # type: np.ndarray
speedo[speedo < 150] = 0


def launchcam():
    mycam = cv2.VideoCapture(0)
    for i in range(100):
        success, frame = mycam.read()
        print(i, frame.shape)
        cv2.imshow("MyCam", process(frame))
        cv2.waitKey(10)


def process(frame):
    out = frame.copy()
    frX, frY, _ = frame.shape
    spX, spY, _ = speedo.shape
    spmask = np.nonzero(speedo)
    out[frX-spX:, :spY][spmask] = speedo[spmask]
    return out


def rotate_thing():
    speedin = cv2.imread(projectroot + "speedin.jpg")
    speedout = cv2.imread(projectroot + "speedout.jpg")
    cv2.rotate()


if __name__ == '__main__':
    launchcam()
