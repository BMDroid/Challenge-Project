import os
import sys
import cv2

def pre_process(resized):
    blur = cv2.GaussianBlur(resized, (1, 1), 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return grey

def is_detected(grey, cascade):
    cards = card_cascade.detectMultiScale(grey, 1.006, 3)
    if len(cards) >= 1:
        return True, cards
    return False, None

def ground_truth(fileName):
    # store the ground truth bounding box position (x, y, w, h) of each file in dictionary
    dic = {}
    with open(fileName) as f:  
        for line in f:
            box = line[31:].split()
            dic[line[:28]] = list(map(int, box))
    return dic

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
    # specify the folder name of the test sets
    folderName = './test'
    dic = ground_truth(f"{folderName}/test.txt")

    # calculate the error rate of test set
    count = 0
    c = 0
    missClassified = 0
    for fileName in os.listdir(folderName):
        # load the image
        if fileName[-3:] == 'jpg':
            img = cv2.imread(f"{folderName}/{fileName}")
            # flag the ground truth of the test images based on their name
            flag = True if fileName[:3] != 'neg' else False
            grey = pre_process(img)
            detected, cards = is_detected(grey, card_cascade)
            if flag and detected:
                ious = []
                x1, y1, w1, h1 = dic[fileName]
                boxA = (x1, y1, x1+w1, y1+h1)
                for (x, y, w, h) in cards:
                    boxB = (x, y, x+w, y+h)
                    iou = bb_intersection_over_union(boxA, boxB)
                    ious.append(iou)
                if max(ious) < 0.5:
                    missClassified += 1
            elif flag != detected:
                if not flag and detected:
                    c += 1
                missClassified += 1
            count += 1
    print(c)
    errRate = missClassified / count * 100
    print("The error rate of the classfier is {0:.3f}%.".format(errRate))
