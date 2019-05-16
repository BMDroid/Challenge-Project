import cv2
import os
import sys

def img_resize(folderName):
    ''' Resize images to certain ratio.
    Args:
        folderName::str
        The name of the folder stores the images.
    Returns:
        None
    '''
    for fileName in os.listdir(folderName):
        print(f"./{folderName}/{fileName}")
        img = cv2.imread(f"./{folderName}/{fileName}")
        resized_image = cv2.resize(img, (288, 180))
        cv2.imwrite(f"../pos_resize/{fileName}", resized_image)

if __name__ == '__main__':
    folderName = '../pos'
    img_resize(folderName)
