# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/test_data_barchart.png "Train Data Distribution"
[image2]: ./images/test_data_barchart.png "Validation Data Distribution"
[image3]: ./images/test_data_barchart.png "Test Data Distribution"
[image4]: ./traffic_signs_data/20mph_zone.png "20 mph Zone"
[image5]: ./traffic_signs_data/70mph_zone.png "70 mph Zone"
[image6]: ./traffic_signs_data/100mph_zone.png "100 mph Zone"
[image7]: ./traffic_signs_data/caution_roadworks.png "Caution Roadworks"
[image8]: ./traffic_signs_data/turn_right_ahead.jpg "Turn Right Ahead"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/johnadams2076/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a set of bar charts showing how the train, validation and test data are distributed.

![alt text][image1]

![alt text][image2]

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


![alt text][image2]

I normalized the image data so as to have mean zero and equal variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |   									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 = 400	            | 
| Fully connected		| outputs 120                                   |
| RELU	                |                                               |
| Fully connected		| outputs 84                                    |
| RELU	                |                                               |
| Softmax				| outputs 43        						    |
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with batch size 128, 200 epochs and 0.001 as learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.934
* test set accuracy of 0.920

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  Lots of augmented data and LeNet architecture was used initially. It worked for the pros.
* What were some problems with the initial architecture?
  The validation accuracy was stuck at around 0.877. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  Multi-scale features was used instead of simple feed-forward.
  Dropouts and an extra fully connected layer was added.
  None of these techniques helped improve the validation accuracy. However, validation and test accuracy both were at around 0.877.  
 Finally, reverted back to LeNet and basic datasets.
* Which parameters were tuned? How were they adjusted and why?
  learning rate was tried with logarithm increments. Sample rates such as 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001.
  Convergence occurred with 0.001, 0.0001. With un-augmneted data 0.001 was faster.   
  Batch sizes tried included, 16, 32, 64 and 128.
  Number of Epochs tried included, 100, 200, 500, 1000, and 2000.  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  Convolution layers fasten the process for image classification. CNNs work on parts of the image making it more practical to work with. Dropouts really did not help in improving the accuracy.
If a well known architecture was chosen:
* What architecture was chosen?
  LeNet
* Why did you believe it would be relevant to the traffic sign application?
  LeNet was developed for image classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there is a watermark across the sign.
Second image has blue sky and greenery in the background. Third image is at a banking angle. Fourth image has tree trunk, sky and greenery.
Fifth image has other signs in the background. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/hr      		| Speed limit (60km/hr) 						| 
| Stop     			    | Speed limit (60km/hr)   						|
| Yield		            | Speed limit (60km/hr)   						|
| No Vehicles      		| Speed limit (60km/hr)    					 	|
| Priority Road		    | Speed limit (60km/hr)         				|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 45th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

[9.8548776e-01 7.5418502e-03 3.3757300e-03 2.3093659e-03 6.0939702e-04]
[3 5 9 25 33]
For the second image ... 

[9.8038059e-01 1.1434657e-02 3.9610234e-03 2.4156836e-03 6.9850113e-04]
[3 5 9 25 7]

For the third image ... 

[9.7218770e-01 1.8986085e-02 3.8553888e-03 2.2549105e-03 1.2996516e-03]
[3 5 25 9 7]
For the fouth image ... 

[9.8385608e-01 6.1478778e-03 6.0918378e-03 2.5152806e-03 4.8459703e-04]
[3 9 5 25 7]
For the fifth image ... 

[9.8326010e-01 1.0538182e-02 2.5541489e-03 2.2845601e-03 6.7034567e-04]
[3 5 9 25 7]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


