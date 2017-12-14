
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

[image1]: ./examples/data_visual.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/_67626131_speed-limit.jpg "Speed limit (120km/h)"
[image5]: ./examples/traffic-signs-achtung-unfallschwerpunkt-german-for-warning-accident-CRDR2P.jpg "General caution"
[image6]: ./examples/100_1607.jpg "Right-of-way at the next intersection"
[image7]: ./examples/Do-Not-Enter.jpg "No entry"
[image8]: ./examples/Arterial.jpg "Priority road"
[image9]: ./examples/conv1.png "Network visualization for the first convolutional layer"
[image10]: ./examples/conv.png "Network visualization for the second convolutional layer"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples.
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43 classes.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. We can see that number of training examples for different traffic signs vary significantly which will probably lead to different accuracies of individual signs prediction.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Regarding preprocessing I did grayscaling and normalization. Grayscaling was done because I considered that all the information of the traffic sign is contained in the black and white version and the color is basically just noise for the network so grayscaling can be considered as noise filtering.

As the last step I normalised the data to remove the ofset because it carries no information and if we keep it we are forcing the network to learn it which will reduce validation accuracy.

No aditional data was genereted because the network results were well within project specifications.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers serialy connected one after the other:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 B&W image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 24x24x16 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 20x20x32 	|
| RELU					|												|
| Max pooling    		| 2x2 stride,  outputs 10x10x32					|
| Fully connected		| input 3200, output 120						|
| Fully connected		| input 120, output 84							|
| Fully connected		| input 84, output 43							|
| Softmax				|            									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using Adam optimizer to minimize cross entropy mean computed using the softmax_cross_entropy_with_logits() function from TensorFlow's nn module. For learning rate, batch size and number of epochs parameters values from LeNet implementation were used. I attempted to use various number of training epochs: 15, 20 and 30 but all of them althoug producing better validation accuracy values lead to overtraining of the network. This showd that the network could not correctly classify all of the test examples found on the internet. Most common missclassification occured with the Speed limit (120km/h) sign that would be classified as Speed limit (20km/h) or as Wild animals crossing sign. When 10 training epochs were used the network showed lower validation accuracy but all the test signs were correctly classified.  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.960 
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

    For the first architecture LeNet architecture was used as a starting point.


* What were some problems with the initial architecture?

    First architecture by itself could not reach the required 0.93 validation set accuracy. After applying grayscaling and normalization of the dataset and raising a number of epochs to 20 the network could intermittently reach 0.93 mark on validation dataset.


* How was the architecture adjusted and why was it adjusted?

    In order to have a reach a higher accuracy I tried to add more layers into the network. First I added one more convolution - relu - max pooling layer with adjusting the kernel sizes in other layers but that lead to even lower validation accuracy result. After that inspired by AlexNet and VGG I removed pooling layers between convolutions leaving just the one before fully connected layer. This lead to very good results.


* Which parameters were tuned? How were they adjusted and why?

    Only parameters that were adjusted were the Kernel sizes because the new convolution layer was introduced and pooling layers removed.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    Two important design choices were adding one aditional convolution layer with RELU activation function and removing all the pooling layers except for the last one. Keeping the original pooling layers inbetween convolutional layers resulted in a too big information reduction for an 32x32x1 input image.
 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All images were cropped to a size of 32x32 centered on the traffic sign. Pictures are very clear and visible apart from one having a watermark over it but I considered it insignificant for the purpouse of sign detection.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        | Prediction	        					    | 
|:-------------------------------------:|:---------------------------------------------:| 
| Speed limit (120km/h)                 | Speed limit (120km/h)  					    | 
| General caution                       | General caution 				                |
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| No entry	      		                | No entry					 				    |
| Priority road			                | Priority road            						|



The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.6%. The model had some trouble in predicting the "Speed limit (120km/h)" sign due to it being similar with other speed limit signs. We will see this in the next section.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model very narrowly made the correct prediction and we can see that the difference is less than 1 percent. The signs are quite similar but it's interesting that the probability of the sign being "Speed limit (20km/h)" is according to the model lower that the probability that the sign is "Speed limit (80km/h) although the first one is more similar to the input. This also leads us to believe that the model is not so reliable in detecting various instances of "Speed limit (120km/h)" sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 25.47%       			| Speed limit (120km/h)         				| 
| 24.79%   				| Speed limit (80km/h) 		     				|
| 21.66%				| Speed limit (20km/h)							|
| 16.76%    			| Stop      					 				|
| 16.75%			    | Speed limit (70km/h)    						|

For the second image the model is quite certain that the image is "General caution" and the margin of difference between other predictions is quite large.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 61.45%      			| General caution								| 
| 25.86%  				| Go straight or left		         			|
| 25.31%				| Traffic signals								|
| 22.12%      			| Turn right ahead	         	 				|
| 21.03%			    | Right-of-way at the next intersection			|

For the third image the model detected correctly "Right-of-way at the next intersection" sign and the difference between other predictions is also quite big.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 41.72%      			| Right-of-way at the next intersection 		| 
| 29.43%  				| Beware of ice/snow							|
| 21.19%				| Vehicles over 3.5 metric tons prohibited		|
|  7.5%      			| Double curve					 				|
|  6.75%			    | Children crossing      						|

For the fourth image prediction probability is the lowest of the whole test set. But is quite ok when we take into account that there are 43 different classes and that the diference between other predictions is quite big.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 19.13%      			| No entry   									| 
|  6.64%  				| Stop  										|
|  6.26%				| Speed limit (30km/h)							|
|  4.49%      			| Turn right ahead				 				|
|  3.01%			    | Keep right      	     						|

The forth image was correctly predicted that it is "Priority road" with 29.03%.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 29.03%      			| Priority road   								| 
| 10.87%  				| Roundabout mandatory 							|
|  5.86%				| No entry			     						|
|  5.72%      			| No passing					 				|
|  5.43%			    | Speed limit (100km/h)    						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image9]
![alt text][image10]

It the image above we have visualizations of the first and second convolutional layer outputs for the "Speed limit (120km/h)" on the input. On these visualizations we can see how the network choses different characteristics of the photo for the feature maps of the first convolutional layer and how this later propagates on the feature maps in the second layer. In the first layer we can see how the network focuses on the various aspects of the sign such are the numbers, area around the numbers, sign border, shape, and how it is for example attempting to invert sign colours. For example in the Feature Map 5 for the layer 1 we have in a way a photo negative of the sign while in Feature Map 4 the network focuses on the border areas between colours which gives the sign a 3D appearance. In the second layer we can see how the network continues to focus on different parts of the image this time derived from the first layer.
