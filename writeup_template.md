# **Traffic Sign Recognition** 

## Image Classification Writup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[Sign1]: ./test_images/1.png "Traffic Sign 1"
[Sign2]: ./test_images/2.png "Traffic Sign 2"
[Sign3]: ./test_images/3.png "Traffic Sign 3"
[Sign4]: ./test_images/4.png "Traffic Sign 4"
[Sign5]: ./test_images/5.png "Traffic Sign 5"
[Sign6]: ./test_images/6.png "Traffic Sign 5"
[exm1]: ./examples/17a.png "example Sign"
[exm2]: ./examples/17b.png "example processed"
[exm3]: ./examples/17c.png "example rotate"
[barchart]: ./examples/TrainingSetDistribution.png "Training set Distribution"
[barchart2]: ./examples/TrainingSetDistribution-AUG.png "Training set Distribution"

[sm1]: ./examples/SM1.png "SoftMax 1"
[sm3]: ./examples/SM3.png "SoftMax 2"
[sm12]: ./examples/SM12.png "SoftMax 3"
[sm13]: ./examples/SM13.png "SoftMax 4"
[sm25]: ./examples/SM25.png "SoftMax 5"
[sm36]: ./examples/SM36.png "SoftMax 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The Project Write-up and the rest ofthe projecty code can be found on my Github, link here -->  [project report code](https://github.com/TrunksLegendary/Udacity-SDCE-CarND-Traffic-Sign-Classifier-Project/blob/master/Udacity-SDCE-CarND-Traffic-Sign-Classifier-Project.ipynb)

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The pandas library ued to calculate summary statistics of the traffic
signs data set, revealed the following details:

* The size of training set: 34799
* The size of validation set: 4410
* The size of test set: 12630
* The shape of a traffic sign image: 32x32x3 
* The number of unique classes/labels in the data set: 43

---
#### 2. An exploratory visualization of the Training dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the 43 traffic sign classes.

The distribution is very uneven with many classes that are underrepresented.

![bar][barchart]

### The following is an exploratory visualization of the Augmented Training dataset.

Here is another exploratory visualization of the same training data set once it has been augmented. This chart shows the distribution of the 43 augmented traffic sign classes.

* The size of training set increased by : 16891
* The new total number of training images : 51690

![bar][barchart2]

---
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, the image is converted  to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling. The inclusion of color did not improve the model performance,  the model actually performed better in grayscale. Also, the lower dimension of the images helped train the model faster.

![alt text][exm1] ![alt text][exm2]

As a last step, the image data was normalized in order to prevent my gradient from going out control in such wide pixel range. This will help avoid my model getting stuck in local minima when I use a single, global learning rate.

The images were loaded from pickled data divided into training, validation, and testing data. The number of images were 34799, 4410, and 12630, respectively. 

Additional data was generated because many classes are significantly under-represented as seen in the distriution plot above. To add more data to the the data set, images with random rotations were added.

Here is an example of an original image and an augmented image:

![alt text][exm1] ![alt text][exm3]

The augmented data was useful, as I saw a a significant increase in accuracy, although there was a high overhead to computation time requirements. Additional work will be needed to optimize this code.

---
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image                       | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 6x6x64 	|
| Dropout               | keep_prob = 0.5                               |
| Max pooling	      	| 2x2 stride,  outputs 3x3x64                   |
| RELU					|                                               |
| Flattening            | outputs 576                                   |
| Fully connected		| outputs 240                                   |
| Dropout               | keep_prob = 0.5                               |
| RELU					|												|
| Fully connected		| outputs 168                                   |
| Dropout               | keep_prob = 0.5                               |
| RELU					|												|
| Fully connected		| outputs 84                                    |
| Dropout               | keep_prob = 0.5                               |
| RELU					|												|
| Softmax				| outputs 43                                    |
 

---
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

 The adam optimizer was used on the model with a batch size of 128, 35 epochs, learning rate of 0.0005, and keep probability of 0.5. Numerous hyperparemeters were tested and analyzed by plotting the loss and accuracy results. The final parameters were chosen which provided the best result while at the same time not over or underfutting the dataset.

---
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.8
* validation set accuracy of 97.1
* test set accuracy of 83.33

I used the Lenet-5 architecture. The Lenet-5 architecture works well with classification in images because it starts with simpler features and moves on to classify more complex features as the input goes through the layers. While starting with grayscale conversion, normalization, and shuffling of the dataset, the model resulted in around 91.7% validation accuracry. Ajusting some of the hyperparameters brought that up to 97%.1 but the model showed overfitting soon as the validation accuracy was significantly lower than training accuracy. This method shows weakness as I could not improce results even while tuning the parameters and trying different methods. I did not want to risk further 'overfitting'.
 
---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][Sign1] ![Traffic Sign 2][Sign2] ![Traffic Sign 3][Sign3] 
![Traffic Sign 4][Sign4] ![Traffic Sign 5][Sign5] ![Traffic Sign 6][Sign6]

I would expect the model to have a hard time classifying the sixth sign, which is for "Priority Road Work", as the landscape behind the sign on the lower part may blend into the sign itself.

---
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Number    | Image		        |     Prediction        | 
|:---------:|:-----------------:|:---------------------:| 
| 13        |Yield                 | Yield                 | 
|  1        |Speed limit (30km/h)  | Speed limit (30km/h)  |	
|  3        | Speed limit (60km/h)  | Speed limit (60km/h)  |	
| 25        |Road work	      		| ** Wild animals crossing ** |  
| 36        |Go straight or right  | Go straight or right  |   
| 12        |Priority Road         | Priority Road         |  



The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of ...

---
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


Correct Prediction:

![Softmax Image 13][SM13]

Correct Prediction:

![Softmax Image 1][SM1]

Correct Prediction:

![Softmax Image 3][SM3]

### Incorrect:
The correct Number should be # 25

![Softmax Image 25][SM25]

Correct Prediction:

![Softmax Image 36][SM36]

Correct Prediction:

![Softmax Image 12][SM12]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


