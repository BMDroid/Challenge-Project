# Challenge-Project

1. Found a paper "Rapid Object Detection using a Boosted Cascade of Simple Features" . And OpenCV has a trainer for Haar-cascade detection. This is quite promising comparing to deep learning methods since it does not need many positive images with the help of opencv_createsamples.

2. By reading the tutorial online, we first need to build a **negative dataset** which contains the image **without the target object**  in it.  And for our project we need to detect the credit card on different surface with different lighting condition. Thus, I choose the [Floor](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03366823) and [Fabric](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03309808) from the ImageNet to be our negative images.  Sample image is showed below:

<p align="center">
  <img width="500" height="370" src="https://raw.githubusercontent.com/BMDroid/Netvirta-Challenge-Project/master/resources/images/sampleFloor.jpg">
</p>

3. The python script for downloading the image [imgDownload.py](https://github.com/BMDroid/Netvirta-Challenge-Project/blob/master/src/imgDownload.py) is in the "src" folder. And the previous image after transformed to **grayscale** and **resize** to 200 * 200 is showed below:
<p align="center">
  <img width="200" height="200" src="https://raw.githubusercontent.com/BMDroid/Netvirta-Challenge-Project/master/resources/images/transformedFloor.jpg">
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
  <img width="284" height="178" src="https://raw.githubusercontent.com/BMDroid/Netvirta-Challenge-Project/master/resources/images/creditCard.jpg">
</p>

6. Since all the credit card images are in different ratio, for future sample creating and model training, all the credit card images had been resized to **288 * 180** by using the [imgResize.py](https://github.com/BMDroid/Netvirta-Challenge-Project/blob/master/src/imgResize.py) And all the resized images are stored in "pos_resize" folder. After deleting vertical credit card images and opposite side of the credit card image, now we have **73** credit card images.

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
    $ opencv_createsamples -img pos_resize/pos_01.jpg -bg neg/neg.txt -info samples/samples_{img[-6:-4]}.txt -pngoutput samples -num 128 -maxxangle 0.3 -maxyangle 0.3 -maxzangle 0.3 -bgcolor 255 -bgthresh 8 -maxidev 40 -w 48 -h 30
    ```
    By using the [createSamples.py](https://github.com/BMDroid/Netvirta-Challenge-Project/blob/master/src/createSamples.py) we could get all **9344** sample imgages and their descrption file in "samples" folder.
    
9. Then we create the vec file for the postive images we just created.
    ```shell
    $ cd samples
    $ cat samples*.txt > samples.txt
    $ cd ..
    $ opencv_createsamples -info samples/samples.txt -bg neg/neg.txt -vec pos.vec -num 9344 -w 48 -h 30
    ```
    We finally get all files we need for train the model.

