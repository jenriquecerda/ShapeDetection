import cv2
import numpy as np


color = {
    "L-H": [0, 255],
    "L-S": [156, 255],
    "L-V": [144, 255],
    "U-H": [111, 255],
    "U-S": [255, 255],
    "U-V": [255, 255]
}


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def create_trackbar(name, color):
    cv2.namedWindow(name)
    for key, val in color.items():
        cv2.createTrackbar(
            key, name, color[key][0], color[key][1], lambda _: _)


def create_mask_from_trackbar(hsv, name):
    l_h = cv2.getTrackbarPos("L-H", name)
    l_s = cv2.getTrackbarPos("L-S", name)
    l_v = cv2.getTrackbarPos("L-V", name)
    u_h = cv2.getTrackbarPos("U-H", name)
    u_s = cv2.getTrackbarPos("U-S", name)
    u_v = cv2.getTrackbarPos("U-V", name)

    lower_value = np.array([l_h, l_s, l_v])
    upper_value = np.array([u_h, u_s, u_v])

    return cv2.inRange(hsv, lower_value, upper_value)


def get_contours(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def get_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_mask = create_mask_from_trackbar(hsv, "Color")
    inverse_mask = cv2.bitwise_not(color_mask)

    kernel = np.ones((5, 5), np.uint8)

    return cv2.erode(inverse_mask, kernel)


def draw_rectangle(frame, coordinates):
    cv2.rectangle(frame, (coordinates[0], coordinates[1]), (
        coordinates[0]+coordinates[2], coordinates[1]+coordinates[3]), (0, 255, 0), 2)


def draw_text(frame, text, coordinates):
    cv2.putText(
        frame, text, (coordinates[0], coordinates[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))


capture = cv2.VideoCapture(0)

create_trackbar("Color", color)

while True:
    _, frame = capture.read()
    mask = get_mask(frame)
    contours = get_contours(mask)

    if len(contours) > 0:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(
                cnt, 0.025*cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            draw_rectangle(frame, [x, y, w, h])
            if area >= 900 and area <= 1500:
                if len(approx) == 8:
                    draw_text(frame, "Nut", [x, y])
            elif area > 26000 and area:
                if len(approx) <= 5:
                    draw_text(frame, "Closed Tool", [x, y])
                elif len(approx) > 5:
                    draw_text(frame, "Open Tool", [x, y])
    else:
        draw_rectangle(frame, [50, 50, 500, 300])
        draw_text(frame, "No Process", [50, 50])

    imgStack = stackImages(0.8, ([frame, mask]))
    cv2.imshow("Stack", imgStack)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
