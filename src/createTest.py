import os
import sys
import time


def create_samples(folderName):
    for fileName in os.listdir(folderName):
        os.system(f"opencv_createsamples -img {folderName}/{fileName} -bg neg/neg.txt -info test/test_{fileName[-6:-4]}.txt -pngoutput test -num 7 -maxxangle 0 -maxyangle 0 -maxzangle 0 -bgcolor 255 -bgthresh 8 -maxdev 40 -w 48 -h 30")
        time.sleep(1) 

if __name__ == '__main__':
    folderName = 'pos_resize'
    create_samples(folderName)