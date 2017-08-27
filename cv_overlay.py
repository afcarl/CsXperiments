import os
import cv2
import numpy as np

projectroot = os.path.expanduser("~/SciProjects/Project_FIPER/")

speedin = cv2.imread(projectroot + "speedin.jpg")  # type: np.ndarray
speedout = cv2.imread(projectroot + "speedout.jpg")  # type: np.ndarray

ishx, ishy, ishz = speedin.shape
oshx, oshy, oshz = speedout.shape
trX, trY = (oshx - ishx) // 2, (oshy - ishy) // 2


def kmph_stream():
    velocity = 0.
    while 1:
        velocity *= 0.95
        velocity += np.random.uniform(0.0, 10.0)
        yield int(velocity)


def cam_stream():
    # noinspection PyArgumentList
    dev = cv2.VideoCapture(0)
    while 1:
        succes, frame = dev.read()
        if succes:
            yield frame
        else:
            print("Frameskip!")


def get_speedometer(kmph):
    if kmph >= 240:
        kmph = 240
    deg = 30 - kmph
    rotM = cv2.getRotationMatrix2D((ishx//2, ishy//2), deg, 1)
    rotIn = cv2.warpAffine(speedin, rotM, (ishx, ishy))
    mat = speedout.copy()
    mat[trX:trX+ishx, trY:trY+ishy] = rotIn
    return mat


def decorate_frame(frame, kmph):
    speedometer = get_speedometer(kmph)
    output = frame.copy()
    # mask = np.where(speedometer == 0)
    output[-oshx:, :oshy] += speedometer
    cv2.putText(output, "{} km/h".format(kmph), (20, 20), 1, 1, (255, 255, 255))
    return np.clip(output, 0, 255)


def main():
    speeds = kmph_stream()
    camframes = cam_stream()
    for speed, frame in zip(speeds, camframes):
        deco = decorate_frame(frame, speed)
        cv2.imshow("DECORATED", deco)
        cv2.waitKey(10)


if __name__ == '__main__':
    main()
