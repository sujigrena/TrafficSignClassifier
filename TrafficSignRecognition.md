# **Traffic Sign Recognition** 

## Writeup


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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is gathered for different input classes. As we know there are 43 different classes, and the number of images in each input class varies as shown in the histogram. 

![Histogram][./Histogram.png] "Histogram"


As we could see the number of images in different classes are very different. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the processing a coloured images 3 times the computation performed for a grayscale image also here the processing is to be done independent of the color of the images. For instance an old stop sign may seem less red than a stop sign installed recently. Hence to process all the images just by the features not limited to color features, I have converted the color image to grayscale image. Chopping the 3 feature maps reduces the computation also improves accuracy by creating a model independent of the color of the images.

Grayscaling the image consisted of taking a specific portion of the 3 color channels and combining them together. Simply averaging and extracting the color components seemed to produce sharp images. Hence I have used an accepted value which is usually used in image conversion algorithms internally. The portion of red used was 0.2989, the portion of green component used was 0.5870 and the portion of blue component used was 0.1140.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale][./Grayscale.png] "Grayscale"

But the grayscaled imaged seems vibrant for the images captured in day light than the image captured in low light. Since the data set is vast and consisted of images of different light and whether conditions, I normalized the image to create a more or less similar image for any type of image given. This gives a data set that is more or less the same except for the unique features in them and helps the neural network to capture the unique features rather than the light variations in the image. 

Below is the image after normalizing.

![Normalized Image][./NormalizedImage.png] "Normalized Image"


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers which is slightly varied version of LeNet architecture by adding measures to reduce over fitting:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   					| 
| Convolution 5x5   	| 1x1 stride, same padding, outputs 28x28x6     |
| RELU					| 												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU                  |												|
| Max Pooling           | 2x2 stride, outputs 5x5x16 					|
| Fully connected 1		| Input: 400, Output: 120 						|        		
| RELU				    | 												|
| Drop Out 1			| Keep Prob: 0.7								|
| Fully connected 2		| Input: 120, Output: 84 						| 
| RELU				    | 												|
| Drop Out 2			| Keep Prob: 0.8								|
| Fully connected 3		| Input: 84, Output: 43 						| 
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used different hyper parameters. After experimenting with different parameters here are the optimal hyper parameters I identified.

* EPOCHS: 50

* BATCH SIZE: 128

* LEARNING RATE: 0.005

* OPTIMIZER USED: ADAM OPTIMIZER 

I didn't change the optimizer and batch size because Adam optimizer is one of the best optimizers available and the batch size more or less used every where is 128.

The hyper parameters which I used and found inappropriate was,

**Model1:** Epochs: 10, Learning rate: 0.001 
The number of epochs in this case was not enough and the validation accuracy was in the range 89 here. So it shows the model needed some more training.

**Model2:** Epochs: 20, Learing rate: 0.001
The number of epochs in this case also was not enough and here the validation accuracy was around 91. This also implies the model needed more training

**Model3:** Epochs: 100, Learning rate: 0.001
Now the validation accuracy reached was 94. But it took quite more number of epochs to attain 94. So I then changed the learning rate and trained the model again.

**Model4:** Epochs 100, Learning rate: 0.005
Since the accuracy was attained after many epochs I increased the learning rate. This time the accuracy reached 94 but seemed to be saturated after 40 epochs. Training again and again could cause over fitting so I retrained.

**Model 5:** Epochs: 50, Learning rate: 0.005
Now the accuracy attained was 94.9 and seemed the best out of the 5 models.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1
* validation set accuracy of 94.9
* test set accuracy of 93.5

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

LeNet was the architecture I chose and used. It is one of the very accurate and efficient architecture to classify images in the MNIST data set. Hence I adapted the same to classify the traffic sign images.

* What were some problems with the initial architecture?

The only problem I observed was the over fitting or underfitting issues that could raise with the hyper parameters chosen. The LeNet architecture didn't seem to address these issues.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

When I trained my model using the LeNet architecture as such the validation accuracy was around 93%. But when I tried to test the model with test data set the accuracy was around 81%. This implied the model is over trained. To reduce this I added two drop out layers after the fully convolutional layer 1 and layer 2. The drop outs used here in the first layer had a keep probability of 0.7 and the next drop out layer used had a keep probability of 0.8. The drop down was added only after flattening to ensure we don't lose features in the convolution filters. The drop outs are added after activation function just to given the expected values to the next layer but just to reduce the count of the same.

* Which parameters were tuned? How were they adjusted and why?

The parameters I tuned are the epochs and the learning rate. Experimenting with different hyper parameters , models , their loss and accuracy gave me the optimal hyper parameters.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The main idea to choose convolutional neural network over conventional neural network is the scalability and computation efficiency. As we know, even the very small images for instance, with size 32x32 will give 1024 input parameters. When we process pictures of high resoultion this might shoot up very fast. Number of input parameters implies the number of tunable parameters. Tuning very large number of parameters efficiently is very expensive sometimes practically impossible leading to non scalable solution.

One more important thing to note is, the images are meaningful when they are arranged the way they are. Fully connected structure would ignore the spatial information and consider the image as a liner model which is not efficient or even change the way images are interpreted. 

As we are dealing with image classification, to over come the above mentioned problems I choose convolutional neural network.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Speed 30][./ext/174257502.jpg] 
![Speed 50][./ext/images.jpg] 
![Road Work][./ext/shield-319990_1280.jpg] 
![Stop][./ext/stop.PNG]
![No entry][./ext/wsi-imageoptim-French-Road-Signs-No-Entry.jpg]

Out of the 5 images given 3 were classified right.

* The first image, the speed limit 30 was rightly classified because the sign 30 seems to be in the center of the image, and even after resizing the quality of the image was retained without distortion. Hence the model seemed to have predicted it right.

* The second image, speed sign 50 was not classified right. This could be attributed to the fact that, the sign 50 is extended across the width and height of the sign is small compared to the width of the image. This could've not matched with the model which would've learnt to classify a better centered and circular sign.

* The third image, Stop sign was not classified right. This could be attributed to the fact that the resized image seemed distorted. This could've lead to the model not to identify the expected features.

* The fourth image, the no entry sign was classified right. Though the sign is very small compared to the size of the image, the model has classified it right because of the training.

* The fifth image, the Road work was also classified right. This is also because of the training and sign is clear in the input image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h     			| 30 km/h   									| 
| 50 km/h   			| Beware of ice/snow 							|
| Stop Sign				| Speed limit (60km/h)							|
| No Entry	      		| No Entry						 				|
| Road Work				| Road Work		      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 12630 images. The loss might be caused because of the low quality image and less number of training data in few output classes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, Speed limit sign 30 the model is relatively sure that this is a speed limit 30 sign (probability of 0.52), and the prediction is right. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| Speed limit (30km/h)   						| 
| .25     				| Speed limit (50km/h)							|
| .13					| Speed limit (80km/h)							|				
| .001	      			| Speed limit (60km/h)				 			|
| .0005				    | No passing     								|


For the second image, Speed limit 50, the model is very surely predicted that the image is a Beware of ice/snow sign with a probability of 0.98. It is a wrong prediction. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98      		   		| Beware of ice/snow   							| 
| .006     				| Slippery road 								|
| .0003					| Right-of-way at the next intersection			|				
| .0001	      			| Priority road					 				|
| .00002			    | Road work     								|


For the third image, the stop sign, the image again predicts wrongly as Speed limit (60km/h) with a probability of 0.25. The top 5 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .25      		   		| Speed limit (60km/h) 							| 
| .17     				| Speed limit (80km/h)							|
| .009					| Turn right ahead								|				
| .008	      			| No passing				 					|
| .005 				    | Road work    									|

 [17,  9, 33, 41, 35],
 
For the fourth image,No entry sign, the model predicted surely that it as a no entry sign rightly with probability 0.99. The top 5 softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99      		   		| No entry 			 							| 
| .00004   				| No passing									|
| .0000006				| Turn right ahead								|				
| .0000007     			| End of no passing			 					|
| .0000004			    | Ahead only   									|


For the fifth image, the Road work, the model again classified it rightly with exact probability 0.999 which is almost equal to 1. The remaining predictions were very less probabilities in the range e-23

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00e+00      		| Road work			 							| 
| 7.52e-23   			| Right-of-way at the next intersection			|
| 1.03e-25				| Bicycles crossing								|				
| 6.82e-26     			| Double curve				 					|
| 7.66e-27			    | Beware of ice/snow							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I've plotted the trained features of the first and second convolutional layers.I
As we could see the first convolutional layer has extracted the activations from the images as a whole i.e., not looking at the details, getting activated whenever they encounter similar features in the training images.
But the convolutional layer 2 has extracted more abstract features specific to images. We could see that the features such as a diagonal, lines etc are captured in the feature maps of this second layer.
