# What should I expect the data format to be?
The data is images of wheat fields, with bounding boxes for each identified wheat head. Not all images include wheat heads / bounding boxes. The images were recorded in many locations around the world.

The CSV data is simple - the image ID matches up with the filename of a given image, and bounding boxes are provided in COCO format: (x-top left, y-top left, width, height). There is a row in train.csv for each bounding box. Not all images have bounding boxes. 

# What am I predicting?
You are attempting to predict bounding boxes around each wheat head in images that have them. If there are no wheat heads, you must predict no bounding boxes.

# Evaluation Metric - mAP@[IoU=0.5:0.75]

This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.75 with a step size of 0.05. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects. A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object.

The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold.

Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.

# Faster R-CNN 

The architecture of Faster R-CNN is complex because it has several moving parts. The input image of shape ```Height × Width × Depth``` is passed through a ```pre-trained CNN``` up until an intermediate layer, ending up with a convolutional feature map. We use this as a feature extractor for the next part. This technique is very commonly used in the context of Transfer Learning, especially for training a classifier on a small dataset using the weights of a network trained on a bigger dataset.

Next, we have what is called a ```Region Proposal Network (RPN, for short)```. Using the features that the CNN computed, it is used to find up to a predefined number of regions (bounding boxes), which may contain objects.

After having a list of possible relevant objects and their locations in the original image, it becomes a more straightforward problem to solve. Using the features extracted by the CNN and the bounding boxes with relevant objects, we apply ```Region of Interest (RoI) Pooling``` and extract those features which would correspond to the relevant objects into a new tensor.

Finally, comes the ```R-CNN module```, which uses that information to:

* Classify the content in the bounding box (or discard it, using “background” as a label).
* Adjust the bounding box coordinates (so it better fits the object).

# Transfer Learning

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. 

**Scenario-1 (Eg:- mobilenet_v2 contains feature extraction module and classification head seperately)**

```
>>> pretrained_mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
>>> print(pretrained_mobilenet_v2)
>>> backbone = pretrained_mobilenet_v2.features
>>> #print(backbone)
```

**Scenario-2 (Eg:- resnext50_32x4d doest not contain feature extraction module and classification head seperately)**

```
>>> pretrained_resnext50_32x4d = torchvision.models.resnext50_32x4d(pretrained=True)
>>> print(pretrained_resnext50_32x4d)
```

# Resources

https://tryolabs.com/blog/2017/08/30/object-detection-an-overview-in-the-age-of-deep-learning/

https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

https://github.com/Cadene/pretrained-models.pytorch


