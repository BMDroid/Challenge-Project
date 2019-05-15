# Challenge-Project

1. Found a paper "Rapid Object Detection using a Boosted Cascade of Simple Features" . And OpenCV has a trainer for Haar-cascade detection. This is quite promising comparing to deep learning methods since it does not need many positive images with the help of opencv_createsamples.

2. By reading the tutorial online, we first need to build a **negative dataset** which contains the image **without the target object**  in it.  And for our project we need to detect the credit card on different surface with different lighting condition. Thus, I choose the [Floor](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03366823) and [Fabric](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03309808) from the ImageNet to be our negative images.  Sample image is showed below:

<p align="center">
  <img width="500" height="370" src="https://raw.githubusercontent.com/BMDroid/Challenge-Project/master/resources/images/sampleFloor.jpg">
</p>

3. The python script for downloading the image [imgDownload.py](https://github.com/BMDroid/Netvirta-Challenge-Project/blob/master/src/imgDownload.py) is in the "src" folder. And the previous image after transformed to **grayscale** and **resize** to 200 * 200 is showed below:
<p align="center">
  <img width="200" height="200" src="https://raw.githubusercontent.com/BMDroid/Challenge-Project/master/resources/images/transformedFloor.jpg">
</p>

4. Upload the negative images to folder "neg" and use the following commands create the description file for the negative images:

   ```shell
   $ find ./neg -iname "*.jpg" > ./neg/neg.txt 
   ```

   And the *neg.txt* contains:

   ```shell
   $ cat ./neg/neg.txt
   ./neg/neg_0001.jpg                                                      
   ./neg/neg_0002.jpg                                                              
   ./neg/neg_0003.jpg                                                                
   ./neg/neg_0004.jpg                                                               
   ./neg/neg_0005.jpg
   ...
   ```

5. Upload total 76 credit card images (downloaded from Goole Images) as positive images into "pos" folder. The sample image is showed below: 
<p align="center">
  <img width="284" height="178" src="https://raw.githubusercontent.com/BMDroid/Challenge-Project/master/resources/images/creditCard.jpg">
</p>

6. Since all the credit card images are in different ratio, for future sample creating and model training, all the credit card images had been resized to **288 * 180** by using the [imgResize.py](https://github.com/BMDroid/Challenge-Project/blob/master/src/imgResize.py) And all the resized images are stored in "pos_resize" folder. After deleting vertical credit card images and opposite side of the credit card image, now we have **73** credit card images.

7. Then we create the description file contains the postive images path and the loaction and size of the bounding box. Since all the positive images have been resized to same ration, we could easily got the description file with the following command:
    ```shell
    $ find ./pos_resize -name '*.jpg' -exec echo \{\} 1 0 0 288 180 \; > ./pos_resize/pos.txt
    ```
    The pos.txt contains:
    ```shell
    $ cd pos_resize
    $ cat pos.txt
    ./pos_resize/pos_01.jpg 1 0 0 288 180
    ...
    ```
    It means that the pos_01.jpg has one target in it and its position is (0, 0) with the width and hight as (288, 180).
    
8. Then we need to create sample images for future training. OpenCV provides the [opencv_createsamples](https://docs.opencv.org/3.4.1/dc/d88/tutorial_traincascade.html) command for creating the training images. Right now, we have **73** credit card images, thus we need to conduct the following command for each image:
    ```shell
    $ opencv_createsamples -img pos_resize/pos_01.jpg -bg neg/neg.txt -info samples/samples_{img[-6:-4]}.txt -pngoutput samples -num 128 -maxxangle 0 -maxyangle 0 -maxzangle 0 -bgcolor 255 -bgthresh 8 -maxidev 40 -w 48 -h 30
    ```
    
    The sample image created is showed below:
    <p align="center">
      <img width="200" height="200" src="https://raw.githubusercontent.com/BMDroid/Challenge-Project/master/resources/images/sampleImg.jpg">
    </p>
    
    By using the [createSamples.py](https://github.com/BMDroid/Netvirta-Challenge-Project/blob/master/src/createSamples.py) we could get all **9344** sample imgages and their descrption file in "samples" folder. (*All the sample images have been added to samples.7z*)
    
9. Then we create the vec file for the postive images we just created.
    ```shell
    $ cd samples
    $ cat samples*.txt > samples.txt
    $ cd ..
    $ opencv_createsamples -info samples/samples.txt -bg neg/neg.txt -vec pos.vec -num 9344 -w 48 -h 30
    ```
    We finally get all files we need for train the model.

10. By using the following command, the haar feature classifier could be trained:
    ```shell
    $ opencv_traincascade -data output -vec pos.vec -bg neg/neg.txt -numPos 1000 -numNeg 500 -numStages 6 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -featureType HAAR -minHitRate 0.995 -maxFalseAlarmRate 0.2 -w 48 -h 30
    ```
    [The detail of the command could be found here.](https://docs.opencv.org/3.4.1/dc/d88/tutorial_traincascade.html)
    The file for each training stage is stored in "output" folder, and it took 1 hour to train to 6 stages on WSL Ubuntu with 16 Gb ram. 
    To get the final classifier after the training ended, the same command (excpet the **numStages** has been modified to be **6**) has been conducted. And we could get the following output:
    
    <p align="center">
      <img width="400" height="300" src="https://raw.githubusercontent.com/BMDroid/Challenge-Project/master/resources/images/cascade.jpg">
    </p>
    
    And we copy and rename the **"cascade.xml**" to **"cascade6stages.xml""** for future detection usuage.
    
11. After the classissfier was trained, the [cardDetector.py](https://github.com/BMDroid/Netvirta-Challenge-Project/blob/master/src/cardDetector.py) for detecting the card and draw contour is created.

    1. First, we need to resize the image to the similar size of the training samples.
    
        ```python
        def resize(img, scale=0.05):
            # calculate the resize scale
            minLen = min(img.shape[1], img.shape[0])
            scale = 150 / minLen
            width = int(img.shape[1] * scale)
            height = int(img.shape[0] * scale)
            resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
            return resized
        ```
    
    2. Then, I used Gaussian blur and convert the blurred image to the gray.

        ```python
        blur = cv2.GaussianBlur(resized, (3, 3), 0)
        grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ```
      
    3. Detect the largest external contour by using the hierachy of the countour. Then create the rectangle bounding box for the largest 
    contour.
      
        ```python
        # threshold image
        threshed_img = cv2.Canny(resized, 0, 255)
        # find the largest contour and its bounding box
        largestContour = largest_contour(threshed_img)
        lagestPoly = cv2.approxPolyDP(largestContour, 3, True)
        boundRect = cv2.boundingRect(lagestPoly)
        ```
      
    4. Load the classifer and try to detect the card in the image. If detected, print the number of the detected cards.
    
        ```python
        # load the classifier
        card_cascade = cv2.CascadeClassifier('./output/cascade6stages.xml')
        cards = card_cascade.detectMultiScale(grey, 1.005, 5)
        print('Card found: ', len(cards))
        ```
    
    5. For detected cards, computed the iou of the largest contour's bounding box and the detected card bounding box. If ios larger than 0.5, then we say that the largest contour is the contour of the card. Otherwise, the detected contour is chosen to be the contur of the card.
    
        ```python
        for (x,y,w,h) in cards:
          boxA = (x,y,x+w,x+h)
          iou = bb_intersection_over_union(boxA, boxB)
          if iou >= 0.5:
              hull = cv2.convexHull(largestContour)
              cv2.drawContours(resized, [hull], -1, (255, 255, 0), 1)
          else:
              cv2.rectangle(resized, (x,y), (x+w, y+h), (0, 0, 255), 1)
        ```
    6. The result example is showed below:
    
    <p align="center">
      <img width="400" height="300" src="https://raw.githubusercontent.com/BMDroid/Challenge-Project/master/resources/images/result.jpg">
    </p>
