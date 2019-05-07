from urllib.request import Request, urlretrieve
import cv2
import os
import sys
import urllib

def save_images(url):
    '''Download images from Imagenet, transfer the image to grayscale, and resize them.
    Args:
        url::str
        The url for downloading imges in the synset
    Returns:
        Does not return anything.
    '''
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    urls = response.read().decode('utf-8')

    # create new directory "neg" if it does not exit
    if not os.path.exists('neg'):
        os.makedirs('neg')
    
    picNum = 1
    for i in urls.split('\n'):
        try:
            print(i)
            fileName = f"neg/{picNum}.jpg"
            urlretrieve(i, fileName)
            # transform the image to grayscale
            img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            # resize the image
            resized_image = cv2.resize(img, (200, 200))
            cv2.imwrite(fileName, resized_image)
            picNum += 1

        except Exception as e:
            print(str(e))  

if __name__ == '__main__':
    # url from the Imagenet fabric
    # url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03309808'
    # url from the Imagenet floor
    url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03366823'
    save_images(url)