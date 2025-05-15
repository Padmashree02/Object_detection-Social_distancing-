# Application of Object detection

  > Problem - To form a CV system that aids to detect all the people and to each detected person, recognise whether the person is safe or not safe.
    
  > Real-world scenario - Health and safety has become a priority to everyone especially since covid phase. Whether the people are in hospital or clinic, viruses spreads much faster due to they are small they can easily be transmitted.
    
  > In this project the objects are detected on the video (input source), In the video the objects are considered as people. And they are detected using deep neural network that is SSD based model- Mobilenet (known as One stage detector)
    
  > One stage detector :- Effective detectors, that detects the object by analysing the whole image in a single pass.

Solution :-

Network/Layer - ssd_mobilenet

Framework - Tensorflow

Dataset (while training the network) - coco

Library - cv2, numpy, matplotlib.pyplot, moviepy

Pipeline (Inderfence/testing) :-

    : Load the network's model file (stored trained weights). 
    
    : Load the network's config file (stored network's configuration / architexture detail).
  
    : Load and extract the class names from the class file- coco dataset.
  
    : Read the network's files using Caffe framework.
  
    : Read the video and capture the frames (by default an image).
  
    : Store the video frame's dimensions- width and height, frames per second. Create an variable to store the output video
  
    : To each frame of the video, convert the image into blob format- helps the network to understand the image through blob format. 
      Note- the parameters of blob function to convert image into blob is as per the newtwork's configuration text file.
    
    : Set the blob image ready and pass the blob forward to the to the loaded model for detections.
  
    : The result of detection is list of list cardinally with detected classID, confidence score, (x,y) coordinate, width, height.
  
    : Set the detection threshold value- detects how well the object is detected in the detected bounding box.
      Note- Based on model's output, set threshold- 0 (no detection will be display), 0.9 (multiple with irrelevant detections will be display) 
  
    : For each detections, extract the classID, confidence score. 
    
    : Check whether the confidence score satisfies with the detection threshold value. 
    
      - If yes, extract the bounding box dimensions and normalise each dimension wrt to image's shape and store in the "box" list (an empty list)
  
      - Then calculate the center of detected box and store each valid detection's confidence score, box list and calculated centeroid in the "result" list (empyty list)
  
      - Else no, then will pass to the next detection.
  
    : Pass the "result" list for recognizing whether they are safe or not safe.
  
      - Compute the ecuclidean distance between each person, by extracting the center points of all valid detections from the result list and then apply the mathematical operation to get euclidean distance (in the form of matrix)
  
      - From the matrix, check each distance wrt to each detected person's bounding box width whether the distance is less and in case if it is less add them into the "violation" set.
  
      - At the end, with each valid detection draw the bounding box and label each box either as "safe" or "not safe" (by checking if current's detection index exist in the "violation" set).
  
    : Save the output video in defined variable.

Important note :- As the trained weight file's size is beyond github's limit. So, could not be able to upload .pb file, however the .pb file can be access from the SSD's respective github link.
