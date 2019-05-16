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
        os.system(f"opencv_createsamples -img {folderName}/{fileName} -bg ../neg/neg.txt -info ../samples/samples_{fileName[-6:-4]}.txt -pngoutput ../samples -num 256 -maxxangle 0 -maxyangle 0 -maxzangle 0 -bgcolor 255 -bgthresh 8 -maxdev 40 -w 48 -h 30")

if __name__ == '__main__':
    folderName = '../pos_resize'
    create_samples(folderName)