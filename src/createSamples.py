import os
import sys

def create_samples(folderName):
    """ Create sample images for tranning.
    Args:
        folderName::str
        The name of the folder stores the positive images in string.
    Returns:
        None.
    """
    for fileName in os.listdir(folderName):
        os.system(f"opencv_createsamples -img {folderName}/{fileName} -bg neg.txt -info samples/samples{fileName[-6:-4]}.txt -pngoutput samples -num 128 -maxxangle 0.3 -maxyangle 0.3 -maxzangle 0.3 -bgcolor 255 -bgthresh 8 -maxdev 40 -w 48 -h 30")

if __name__ == '__main__':
    folderName = 'pos_resize'
    create_samples(folderName)