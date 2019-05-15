import numpy as np
import cv2
import imutils
import copy

def resize(img, scale=0.05):
    # calculate the resize scale
    minLen = min(img.shape[1], img.shape[0])
    scale = 150 / minLen
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return resized


def largest_contour(img):
    contours, hier = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # Find level 1 contours
    level1 = []
    for i, h in enumerate(hier[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if h[3] == -1:
            h = np.insert(h.copy(), 0, [i])
            level1.append(h)
    # find the contours with large area.
    contoursWithArea = []
    for h in level1:
        i = h[0]
        contour = contours[i]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, i])
    # sort the contours in reverse order
    contoursWithArea.sort(key=lambda x: x[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

def bb_intersection_over_union(boxA, boxB):
	# determine the coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of boxA and boxB
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute iou
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

if __name__ == '__main__':
    # load the classifier
    card_cascade = cv2.CascadeClassifier('./output/cascade6stages.xml')
    
    # load the image
    img = cv2.imread('./test/test1.jpg')
    resized = resize(img)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    height, width = resized.shape[:2]
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # threshold image
    threshed_img = cv2.Canny(resized, 0, 255)

    # find the largest contour and its bounding box
    largestContour = largest_contour(threshed_img)
    lagestPoly = cv2.approxPolyDP(largestContour, 3, True)
    boundRect = cv2.boundingRect(lagestPoly)
    boxB = (int(boundRect[0]), int(boundRect[1]), int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3]))

    # find contours and get the external one
    # contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    cards = card_cascade.detectMultiScale(grey, 1.005, 5)
    print('Card found: ', len(cards))

    for (x,y,w,h) in cards:
        boxA = (x,y,x+w,x+h)
        iou = bb_intersection_over_union(boxA, boxB)
        if iou >= 0.5:
            # cv2.rectangle(resized, (x,y), (x+w, y+h), (0, 0, 255), 1)
            # cv2.rectangle(resized, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (0, 255, 0, 1))
            hull = cv2.convexHull(largestContour)
            cv2.drawContours(resized, [hull], -1, (255, 255, 0), 1)
        else:
            cv2.rectangle(resized, (x,y), (x+w, y+h), (0, 0, 255), 1)

    # create the window
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', width, height)
    cv2.imshow('result', resized)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('test.png',img)
        cv2.destroyAllWindows()