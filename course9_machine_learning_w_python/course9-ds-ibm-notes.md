# Course 9: Machine Learning with Python 

**What you'll learn**

- Describe the various types of Machine Learning algorithms and when to use them  
- Compare and contrast linear classification methods including multiclass prediction, support vector machines, and logistic regression  
- Write Python code that implements various classification techniques including K-Nearest neighbors (KNN), decision trees, and regression trees 
- Evaluate the results from simple linear, non-linear, and multiple regression on a data set using evaluation metrics   

**Skills you'll gain:** Data Science, Big Data
Machine Learning, Deep Learning, Data Mining

## Table of Contents

  - [W1: Introduction to Machine Learning](#w1-introduction-to-machine-learning)
  - [W2: Regression](#w2-regression)
  - [W3: Classification](#w3-classification)
  - [W4: Clustering](#w4-clustering)
  - [W5: Recommender Systems](#w5-recommender-systems)
  - [W6: Final Project](#w6-final-project)
  - [References](#references)


## W1: Introduction to Machine Learning

**Welcome**

- used in many key field industries
	=> healthcare : 
		=> predict cancer level of severity
		=> predict the right drug for a patient according to his age 
	=> Bank : loan application acceptance 
	=> Custumer segmentation 
	=> Recommendation systems (amazon, netflix , googele ..)
	=> Telecommunication : to predict customer churn (?) ...

- Python libs to build ML models
- Jupyter Lab
- Jupyter Notebook for coding/modeling ...

**Skills Insights:** 
- Regression
- Classification
- Clustering
- Scikit Learn
- Scipy ... 

**Projects:** 

- Cancer detection,
- Predicting economic trends
- Predicting customer churn
- Recommendation engines ...

### Introduction to Machine Learning

- (By @Arthur Samuel) : Subfield of computer science that gives "computer the ability to learn without being explicitly programmed" 
- ML helps with predictions 
- ML process : 
	=> dataset (data gathering=), 
	=> data cleaning
	=> select the proper algorithm
	=> building model
	=> training 
	=> make predict (based on high accuracy)

## How ML works ?
- Solving Problem with tradictinal approach  : Animal face detection 
(Input dataset) => (Feature Extraction : vector features ) => (A set of rules) => (Prediction) ? 
	=> Consequences : LOTS of rules => highly dependent on current dataset/tool specific not generalized
	=> Not generalized enough to detect out of sample cases 

- ML APPROACH : 
	=> Allows to build a model that looks at all the feature sets and their corresponding type of objects(animals) and learns the pattern of each target object(animal)
	=> The model is built by machine learning algorithms
	=> The goal is to detect without explicitily being programmed to do so  
- machine learning follow the same process that a 4ys old child uses to LEARN !!!
- Inspired by the human learning process iteratively 
- Learn from data => allow computers to find *hidden insights*
- The model help us in variety of tasks : 
	=> object recognition 
	=> object summarization 
	=> object recommendation ...
## ML Application (real life examples) : 
- Welcome Chapiter!!!
## ML techniques : 
- Regression/Estimation :
	=> Predicting continuous values (house prices, CO2 emission of car engine ...)
- Classification
	=> Predicting the item classy/category of a case  (true/False cases ? )
- Clustering : groups of similar cases (find similar patients, or customer segmentation in the bank)
	=> Finding the structure of data;summarization ... 
- Associations 
	=> Associating frequence co-occuring items/events (grocery items bought together by a customer)
- Anomaly detection :
	=> Discovering abnormal and unusual cases (credict vard detection ...)
- Sequence mining : 
	=> Predicting next events; click-stream (Markov Model, HMM)
- Dimension Reduction
	=> Reducing the size of data (PCA)
- Recommendation sytems 
	=> Recommendation items 

## AI vs ML vs DL 
AI  : try to make computer intelligent trying to mimic the cognitive functions of human
- Computer Vision (cv)
- Language Processing 
- Creativity
...

ML : branch of AI that covers the statistical side of AI ...
- Regression
- Classification
- Clustering
- Neural Network
...

DL : THE REVOLUTION in ML !!!
- allows computers to make decision by their own  

# Python for Machine Learning
- General language for writing ML algorithms
## Python Library : 
- NumPy : math library to work w/ N-dimensional arrays in Python 
	=> make computation effeciently and effectivily 
	=> powerful than regular python(arrays, dicts, functions, datatypes, images ...)
	=> /!\ Linear algebra /!\ 
- Scipy : collection of numerical algorithms and domain specific toolboxes : 
	=> scientific and high perfomance computation
	=> signal processing 
	=> optimization 
	=> statistics ...
- Matplotlib : data visualization (plotting) package
	=> 2D and 3D plotting   	
- Pandas : High level python lib for data processing 
	=> datastructure + (built) functions for manipulating numerical tables, timeseries
	=> /!\ Data Exploratory /!\ 

- Scikit learn  : set of algorithms and tools for ML 
	=> Free sw machine learning library
	=> Regression, Classification and Clustering algorithms
	=> Works with Numpy and SciPy
	=> Great documentation 
	=> Easy to implement (model)

The full Machine leaning pipelines is already implemented in Scikit learn 

```
+-----------------+	 +-----------------+	+------------------+	+-------------+	   +-----------+	+----------+	+------------+
| Data Processing |=>| Train/test split| => | Algorithms setup | => |Model fitting| => |Predictions| => |Evaluation| => |Model export|
+-----------------+	 +-----------------+	+------------------+	+-------------+	   +-----------+	+----------+	+------------+
```

Ref : course 7 : data analysis w/ python

# Supervised vs Unsupervised
- Supervised learning  : teach the model, then w/ that knowledge, it can predict unknown or future instances 
	=> numerical data
	=> categorical data (target data)
	=> Classification & Regression are two types of SP
	=> controlled Environment

- Unsupervised learning : the model works on its own to discover information
	=> learn from a giving dataset and draws conclusion on unlabled data. 
	=> Techniques : 
		=> Dimension Reduction
		=> Density estimation 
		=> Market basket analysis
		=> Clustering (most common) 
	=> Uncontrolled env

# Quiz : Intro to Machine Learning

## W2: Regression

Ref : DATA ANALYIS W/ PYTHON => MODEL 

# Introduction to Regression
- Process of predicting a continuous value
- 2 types of variables in regresssion : 
	=> X : independent (explanatory var) - cause of state/target
	=> Y : dependent(state, target, final goal to study/predict)
- Y : shall be continuous not be discrete value
- X[i] : can be measured on either a categorical ou continuous
## Regression Model : 
(Historical data:one/more features) => (Modeling) => (Prediction)
(new data:) --------------------------------^

## Types of regression models 
- Simple Regression :  single independent var => estimate single dependent variable
	=> Simple Linear Regression (ex: Predict 'co2emission' vs 'EngineSize' of all cars)
	=> Simple Non-Linear Regression (depending on relationship btw X & Y)
- Multiple Regression : more than one independent variables 
	=> Multiple Linear Regression  (ex: Predict 'co2emission' vs 'EngineSize' & 'Cylinders' of all cars)
	=> Multiple Non-Linear Regression depending on relationship btw X & Y
## Applications of Regression 
- Sales forecasting
- Satisfaction analysis 
- Price estimation
- Employement income

## Regression Algorithms : 
- Ordinal regression
- Poisson regression
- Fast forest quantile regression
- Linear, Polynomial, Lasso, Stepwise, Ridge regression
- Bayesian linear regression
- Neural network regression
- Decision forest regression
- Boosted decision tree regression
- KNN(K-nearest neighbors) 

# Simple Linear Regression
- Using linear regression to predict continuous values
## How does linear regression work 
- Linear regression : you can model the relationship of the these variables
- A good model can be used to predict what the approximate Y of each X

## Linear regression model representation
- the regression line represents a good fit of the data 
- can be used to predite an approximated values

- fitting line = polynomial 

```
------------------------
| ^y = a*x + b (model) |
------------------------
```

 ^y : response variable 
 x  : predictor variable 
 ## Adjustable parameters 
 a : slope / gradient of fitting line / coedf of linear equation
 b  : intercept

## How to find the best fit ? 
- To calculate the fitting line : 
- calculate a, b that best fit for the data 

y : a*x + b => real value => i.e for x = value => y = result  ?
^y : estimated depend of fitting line 

```
+-----------------+
| Error = y - ^y  |
+-----------------+
```

- Error : distance from data point to the fitted regression line 
- The mean of all residual errors shows how poorly the line fit with the whole dataset mathematically shown by : 
MSE : Mean Square Error 

```
+-------------------------+
|MSE = 1/n*Sum(Yi - ^Yi)^2| where  : i = 1 ...n
+-------------------------+
```

- Goal : Find a line where the mean of all these errors is minimized
	=> The mean error of the prediction using the fit line should be minimized
- The objective of linear regression is to minimize this MSE eq. 
	=> by finding the best parameters (a, b)
	=> mathematical approach
	=> Optimization approach
## Estimating the parameters :
- mathematical approach
	^y: a*x + b 
	a ? 
	b ? 
	=> this parameters can be calculated using python lib, R, ou scala ... 
	=> rebuild the polynom

- Optimization approach ?

## Pros of linear regression 
- very fast 
- No parameter tunning 
- Easy to understand, and highly interpretable

# Model Evaluation in Regression Models
- The goal of regression is to build a model to accurately predict an unknown case 
- For that regression evaluation has to be done after building the model
## Model evaluation approaches
1/ Train and Test on the Same Dataset 
2/ Train/Test split
- Regression Evaluation metrics

1/ Train and Test on the Same Dataset : How to calculate the accuracy of a model ??
- 70% of dataset for training => training set(feature set) 
- 30% of dataset for testing => testing set 
	=>the labels of test set are called actual value of the test set
	=> actual values are used for 'ground truth'
- At the end we compare the Predict values VS actual values => give the accuracy
- One of the Metrics to calculate the accuracy : 

```
+--------------------------+
|Error = 1/n*Sum(yi - ^yi) |
+--------------------------+
```

- where : i = 1 .. n 

## Training and out-of-sample accuracy? 

- Training Accuracy : percetange of the correct prediction that the model makes when using the test dataset 
	=>(-) High training accuracy isn't necessarily a good thing
	=>(-) Result of over-fitting : the model is overly trained to the dataset, which may capture noise and produce a non-generalized moodel
	=>  
- Out-of-sample(OOS) : percetange of the correct prediction that the model makes on the data the model has not been trained on. 
	=> it's important that our models have a high OOS accuracy (model main goal is to make great prediction mostly on unknown datas)
	=> How can we improve OOS accuracy ? 
		=> Using train/test split approach

2/ Train/Test split : 
- splits the dataset into training/testing data "mutually exclusive"
- train with training set 
- test with testing set
- (+) More accurate evaluation on OOS accuracy (because the testing dataset is not part of the dataset that has been used to train the data)
- training the model with testing data afterwords avoid loosing valuable data
(-) Highly dependent on which datasets the data is trained and tested
	=> Ex. K-fold cross validation solves the issue above
	=> by *averaging the high variation that results from a dependency *

# Evaluation Metrics in Regression Models
- Used to explain the performance of the model 
- Error (regression) : measure of how far the data is from the fitted regression line.

```
+--------------------------+
|Error = 1/n*Sum(yi - ^yi) |
+--------------------------+
```

- Most common Model evaluation metrics :
	=> MAE : Mean Absolute Error 

```
+--------------------------+
|MAE = 1/n*Sum(|yi - ^yi|) |
+--------------------------+
```
	=> MSE : Mean Squared error 

```
+--------------------------+
|MSE = 1/n*Sum(yi - ^yi)2  |
+--------------------------+
```
	=> RMSE : Root Mean Squared error 

```
+--------------------------+
|RMSE^2 = 1/n*Sum(yi - ^yi)|
+--------------------------+
```
	=> RAE : Relative Absolute Error/Residual sum of square 

```
+---------------------------+
|RAE = n*MAE/Sum(|yi - ÿi|) | where ÿ : mean of y 
+---------------------------+
```
	=> RSE : Relative Squared error : (very similar to MAE, is used for Normalization) => used to calculate R^2

```
+--------------------------------------+
|RSE = Sum(yi - ^yi)^2 / Sum(yi - ÿ)^2 |
+--------------------------------------+
```
- R squared  : represents how close the data values are to the fitted regression line 
=> the more R^2 is higher the more the model fitted the data  

[R^2 = 1 - RSE]  

# Lab: Simple Linear Regression

# Multiple Linear Regression
- more than one independent variables 
- ex: Predict 'co2emission' with 'EngineSize' & 'Cylinders' of all cars
- extension of linear regression model  

## Examples of multiple linear regression
- independent variables effectiveness on prediction ( the strengh of the effect of that the independent vars have on the dependent variable)
	=> Does revision time, test anxiety, lecture attendance and gender have any effect on the exam perfomance of students ?
- Predicting impacts of changes (how dependent variables changes when independent variable changes)
	=> How much does blood pressure go up (or down) for every unit increase (or decrease) in the BMI-'Body Mass Index' of a patient

## Predicting continuous values with multiple linear regression
- uses multiple independent variables(predictors) to that best predict the value of the target(dependent) 
- X: independent variables
- Y: dependent variable (linear combination of X)
	=> MODEL : 

```
+------------------------------------------------------+
| ^Y = theta0 + theta1*x1 + theta2*x2 + ...+ thetan*xn |
+------------------------------------------------------+
```
	=> mathematically vector form : multidimensional space
```
+--------------------+
| ^y = (theta^T) x (X) | 
+--------------------+
```

```
    => theta : vector of coeficients - the parameters/weight vector of the regression equation
		=> dim(theta) = n by 1
	=>(theta^T) : [theta0, theta1, theta2...]
	=> T : transpose
	=> ()x(): cross product 
	=> X : vector of featured sets
		[1]
		[x1]
	X = [x2]
		[..]
	=> the first element of the X vector is : 1 => it turns that theta0 into the intercept/biased
```

- enable to evaluate which (independent)variable is significant predictores of the outcome variable
	=> how eah feature impact the outcome variable
- (theta^T)^X : also called plane /hyperplane for multiple regression
- GOAL : find the best fit hyperplane for input data
	=> minimize the error of the prediction
	=> find the optimized parameters

## Using MSE to explose the errors in the model 

- sum of residual errors (actual value - predicted values(from the model))

## Estimating multiple linear regression parameters

- estimate theta ? 
	=> Ordinary Least Squares(Moindre au carré)
		=> Linear algebra operations
		=> Takes a long time for large datasets (10K + rows)
	=> An optimization algorithm
		=> Gradient Descent (starts the optimization w/ randoms values for each coef and compute the errors and try minimize it through Y^s changing of the coef in multiple iterations)
		=> proper approach for very large dataset

## Making prediction w/ multiple linear regression
- making predictions is as simple as solving the equation for specific set of inputs
- from the model : ^y = (theta^T)^X
- compute the coefs [theta0 ... n]
- plug the parameters into the linear model	
	=> replace the independent varibles & the coefs => gives the new dependent value predicted

## Q&A - on multiple linear regression : 
- How to determine whether to use simple or multiple linear regression?
- How many independent variables should you use ? 
- Should the indepent varible be continuous
- What are linear relationships btw the depent variable and the independent variables

# Lab: Multiple Linear Regression

# Non-Linear Regression

- behavior is not described by a straight line
- the goal still to estimate the parameters of the model the use the fitted model to predict the target/labal for unknown/future cases 

## Different types of regression 
- Linear Regression
- Quadratic (Parabolic) Regression 
- Cubic Regression 
...
- all these type of regressions can be called Polynomial regression : Y is modeled as Nth degree polynomial in X.
- Polynomial regression fits a curve line to your data
- a normal polynomial regression can be converted as Multiple linear regression and fit w/ Least Squares
	=> Minimizing the sum of the squares of the differences btw y and ^y
## Non-linear regression features ?
- To model non-linear relationship btw the dependent variable and a set of independent variables 
- ^y must be a non-linear function of the parameters (theta), not necessarily the feature x
- Mathematically : exponential, logarithmic, and logistic etc 
- The model (^y) depends not only on the x values but alson on the coefficients (theta0...thetaN)
- Ordinary Least squares cannot be used to fit the data 


## Check lineat vs Non linear regression 
- inspect visually
- based on accuracy 
	=> if theta > 0.7 ==> linear

- Modeling :
=> Polynomial regression
=> Non-linear regression model 
=> Transform your data 
 
# Lab1: Polynomial Regression
# Lab2: Non-linear Regression
Final Quiz

## W3: Classification

/// K-Nearest Neighbours

# Introduction to Classification
- A supervided learning approach
- Categoring some unknown items into a discrete set of Categories or "classes"
- learn the relationship btw a set of feature variable of interest 
- The target attribute is a categorical variable

## How does classification work ? 
- classification determines the class label for an unlabeled test case

- Modeling Binary Classifier : 

```
						+------------+
	Prediction 		=>	| Classifier | => Predicted Labels (0/1)
(new input values)		+------------+
```

- Same example for multi-classification except we'll have multiple labels

## Applications : 
- email filtering 
- speech recognition
- handwriting recognition
- biometric identification
- document classification ...

## Classification algorithms in ML 
- Decision Trees(ID3, C4.5, C5.0)
- Naïve Bayes
- Linear Discriminant Analysis
- k-Nearest neighbor
- Neural Networks
- Support Vector Machines (SVM)
...

# K-Nearest Neighbours (k-NN)
- Dataset example : 
	=> A Telecommunication provider company has segmented his custumer base by service usage patterns
	=> Categorizing the customers into four groups (1, 2, 3, 4...)
	=> If demografic data can be used to predict group membership 
	=> The company can cutomize offers for individual perperctive customers

- Goal : to predict the new unknown case based on predefined labels ? 
- Dataset structure : 
	=> Xi : independent variables
	=> Yi : dependent variables (groups of classes/labels)
- find the closest cases and assign the same case label to the new values/features

## What i K-Nearest Neighbors - KNN ? 
- classification algorithm that takes a bunch of labeled points and uses them to learn how to label other points
- A method for classifying cases based on their similarity to other cases 
- Cases that are near each other are said to be "neighbors" 
- Based on similar cases w/ same class labels are near each other
- The distance btw two cases is a measure of their dissimilarity
- the distance /dissimilarity btw Two data points be calculated  : 
	=> Euclidean distance 

## How it works ? 
=> the algorithm look for the k-nearest neighbors (class) and do a majority vote among them to define the class of the new value (next case)
=> the k-class is defined based on the number of nearest neighbors class around the case
	1. Pick a value for K 
	2. Calculate the distance of unknown case from all cases.
	3. Select the K-observations in the training data that are "nearest" to the unknown data point
	4. Predict the response of the unknown data point using the most popular response value from the K-nearest neighbors 

## Calculating the similarity/distance in a K-dimensional space 
- calculate the distance of two customers : Euclidean distance using Minkowski distance

```
+--------------------------------------+
| Dis(x1, x2) = (Sum(x1i - x2i)^2)^1/2 |		
+--------------------------------------+
```

where : i=0 ...n, and n is the number of feature

- 2 customers have only one feature : their ages
if n=1, feature1(x1 = 34), feature2(x2 = 30) 
Dis = ((34 - 30)^2)^1/2 = 4 => Class/group : 4
if n=2 ... same approach

## What is the best value of K for KNN ? 
- K and KNN, is the number of nearest neighbors to examine (specified by the user) 
- Choosing the right K ?
	=> the algorithm look for the k-nearest neighbors (class) around the new case and do a majority vote among them to define the class of the new value (new case)
	=> in cas we capture the less voted class => our data is noisy 
	=> low value of K causes a highly complex model (overfitting)
- General solution ?
	=> separate data for testing the accuracy of the model 
	=> choose K=1 use the training part for modeling and calculate the accuracy of prediction using all samples in the testset
	=> the process is repeated increasing the K and see which K is best for the model
## Computing continuous target using KNN
- KNN can also be used for regresion 
- we use the average/median of the NN is used to predicted value for the new case 

# Evaluation Metrics in Classification
- How accurate is the model ? 
	=> y - yhat = ? 
	=> where - y : actual labels
			 - yhat : predicted labels
- Model evaluation helps up improve the model 
=> Jaccard index
=> F1-score
=> Log Loss 

## Jaccard index (Jaccard Similarity coefficient)
- y : actual labels
- yhat : predicted labels

```
+------------------------------------------------------+
| J(y , yhat) = |y intersection yhat| / |y union yhat| |
+------------------------------------------------------+
```
				or
```
+-------------------------------------------------------------------------------+
| J(y , yhat) = |y intersection yhat| / (|y| + |yhat| - |y intersection yhat|)  |
+-------------------------------------------------------------------------------+
```

Ex: labels values:

```
y :    [0,0,0,0,0,1,1,1,1,1]
yhat : [1,1,0,0,0,1,1,1,1,1]
```
```
J(y, yhat) = 8/(10 + 10 - 8) = 0.66
```
- high vs low accuracy 
hight  : J(y, yhat) = 1,0, if all labels matches 
low : J(y, yhat) = 0, othewise  

## F1-score(confusion matrix/heatmap):
- Confusion matrix shows the model ability to correctly predict or separate the classes

```
		   +-------+------+
churn =1   |   TP  | FN   |
		   +-------+------+
churn = 0  |   FP  | TN   |
(predicted)+-------+------+
```
   values  churn=1  churn=0 (actual values)

- in case of Binary classification : the number inside the matrix boxes indicates the count of predicted label 
	=> TP : True Positive 
	=> FN : False Negative 
	=> TN : True Negative 
	=> FP : False Positive

```
+---------------------------+
| Precision  = TP/(TP + FP)	|
+---------------------------+
```

=> true positive rate

```
+-----------------------+
| Recall = TP/(TP + FN) |
+-----------------------+
```

F1-score :

```
+------------------------------------+
|F1-score = 2x(prc x rec)/(prc + rec)|
+------------------------------------+
```

=>where F1-score range [0 ...1], 
0 : closest to 0 low accuracy/worst 
1 : closest to 1 high accuracy/best

## Log loss 
- measures the performance of a classifier where predict output is a probability value btw 0 and 1

```
+--------------------------------------+
|y x log(yhat) + (1 - y)xlog(1 - yhat) |
+--------------------------------------+
```

- measures how far the prediction is from actual label

```
+-----------------------------------------------------------+
| LogLoss = -1/n[Sum(y x log(yhat) + (1 - y)xlog(1 - yhat))] |
+------------------------------------------------------------+
```

LogLoss => Low => Better accuracy 

=>where LogLoss range [0 ...1], 
0 : closest to 0 high accuracy/best 
1 : closest to 1 low accuracy/worst 

/!\ The classifier with lower log loss has better accuracy 

# Lab: KNN

/// Decision Trees
# Introduction to Decision Trees
- example : Medical Dataset about patients suffering from the same illness and each patient was treated different medication : drugs A, B
	=> The dataset has set of feature about the patient : age , sex, BP(Blood Pressure), Cholesterol level, Drug (A, B)
	=> Problem trying to guess which medication(drug) to prescribe to new Patient with the same illness
	=> Class : Drug A, Drug B => binary classification scenario

- Modeling Decision Tree : 

```
						+----------------+
	Prediction 		=>	|  Decision Tree | => Predicted Labels
(new input values)		+----------------+

```


## Building a decision tree with the training set 
- Decision trees are built by splitting the training set into distinct nodes

```
		+-------+
	 	|  Age  |
		+-------+ 
		/   |   \
	  /     |     \ 
 Young 	 mid-age Senior
 	 |		|		  |
	 Sex 	B		Cholesterol
	 / \  	 		 / \
    /   \   		/   \
   F  	 M    	   high Normal
   | 	 |			|	 |
   A 	 B			B	 B
```

- DT is about testing an attribute and branching the cases based on the result of the test 
- Each internal node corresponds to a test (*sex)
- each branch corresponds to a result of the test(*M)
- Each leaf node assigns a classification

## Decision tree learning algorithm
- A decision tree can be constructed by considering the attributes one by one 
1. Choose an attribute from your dataset 
2. Calculate the significance of attribute in splitting of data 
3. Split data based on the value of the best attribute
4. Go to step 1. (repeat)

# Building Decision Trees
- DT are built using recursive partitioning to classify the data 
- The algorithm choose the most predictive feature to split the data on
## Which attribute is the best ? 
	=> More predictiveness
	=> Less Impurity
	=> Lower Entropy 
- Impurity of nodes is calculated by entropy of data in the node 
- Entropy : is the amount of information disorder or the amount of randomness in the data 
	=> The lower the Entropy, the less uniform the distribution, the purer the node

```
 			+-------------------------------------------+
 			| Entropy = - p(A)log(p(A)) - p(B)log(p(B)) |
 			+-------------------------------------------+ 
```

- calculate by the python library 
/!\ The tree with the higher *Information Gain* after splitting 

##  Information Gain
- IG - information that can increase the level of certainty after splitting
- IG opposite of entropy

```
 	+---------------------------------------------------------------------+
 	| IG = (Entropy before split) - (weighted Entropy before after split) |
 	+---------------------------------------------------------------------+ 
``` 

- We start the decision tree with the best attribute

# Lab: Decision Trees

/// Logistic Regression
# Intro to Logistic Regression
- Classification algorithm for categorical variables
- Ex : predict customer churn  
- analogous to linear regression but tries to predict a categorical or discrete target field instead of a numeric one
- Predict a variable which is binary such as yes/no, true/false, successful/not successful, pregnant/not pregnant=> 0/1
- independent variables should be continuous 
	=> for categorical case : dummy/Indicator coded => has to be transformed to some continiuous values

## Application  : 
- use Binary or multi-class classification
	=> Predicting the probability of a person having a heart attack  
	=> Predicting the mortality in injured patients
	=> Predicting a customer's propensity to purchase a product or halt a subscription
	=> Predicting the probability of a failure of a given process or product
	=> Predicting the likelihood of a homeowner defaulting on a mortgage

## When to use  ? 
- If the data is binary (0/1, YES/NO, True/False...)
- If a probabilistic results is needed 
- When a linear decision boundary is needed 
- To understand the impact of a feature 

## Modeling 

- predict the class of each sample ...

- X : independent variable => X E |R ^(mxn) 
- Y : dependent variable => y E {0, 1}

```
+-----------------+
|yhat = P(y=1|x)  |
+-----------------+
```

Where: P : probability 

- Predicting y = 1 given x


```
+--------------------------+
| P(y=0|x) = 1 - P(y=1|x)  |
+--------------------------+
```

# Logistic regression vs Linear regression

- Linear regression : Y = a*X + b = (theta^T x X) 
	=> not good enough for categorical cases 
	=> too error !!!
- Logistic regression

## Sigmoid function 
- Like a step function but less error and gives more information in probability form:

```
+------------------------------------------------+
| Sigma(theta^T x X) = 1/(1 + e^-(theata^T x X)) |
+------------------------------------------------+
```
```
P(y=1|x) => proba increases close to 1 => Sigma(theta^T x X) = 1	 
P(y=1|x) => proba decreases close to 0	=> Sigma(theta^T x X) = 0 
```
```
Sigma(theta^T x X) = [0, 1]
```


## Classifying an specific model 

```
- Churn case : 
	=> P(y=1|x)
	=> P(y=0|x) = 1 - P(y=1|x)

P(y=1|income, age) = 0.8 
P(y=1|income, age) = 1 - 0.8 = 0.2 

Sigma(theta^T x X) 	   => P(y=1 | x) 
1 - Sigma(theta^T x X) => P(y=0 | x) 

```
## Training process 

1. Initialize Theta 
	=> Theta = [-1, 2]
2. Calculate yhat = Sigma(theta^T x X) for a customer
	=> yhat  = Sigma([-1, 2] x [2,2]) = 0.7
3. Compare the output of yhat with actual output of customer, y, and record it as error 
	=> Error = 1 - 0.7 = 0.3
4. Calculate the error for all customers
	=> Cost = J(Theta)
5. Change the Theta to reduce the cost 
	=> NEW Theta
6. Go back to step 2

- when stop training => When the model is satifactory

# Logistic Regression Training
- Goal : change the parameter of the model to be the best estimation of the labels of the samples in the dataset 

Sigma(theta^T x X) 	=> P(y=1 | x) 
- Change the weight => Reduce the cost
- Cost function 
	=> Cost(yhat, y) = 1/2(Sigma(theta^T X) - y)^2 ... ^2 : to avoid negative value 

- Total cost/Mean Squere Error:

```
+------------------------------------+
| J(Theta) = 1/m*Sum(Cost(yhat, y)   | ... where i = 1 ... m 
+------------------------------------+
```
- Find the minimum error value close to desirable estimated value 
- (-log) function fits best  

## Cost function : 

```
Cost(yhat, y) = | -log(yhat) 	 => if y = 1 
				| -log(1 - yhat) => if y = 0 

+------------------------------------------------------------------+
| J(Theta) = -1/m*Sum[y^i*log(yhat^i) + (1 - y^i)*log(1 - yhat^i)] |
+------------------------------------------------------------------+
```

## Minimizing the cost function of the model 
- How to find the best parameters for our model ? 
	=> Minimize the cost function 
- How to minimize the cost function  ?
	=> Using Gradient Descent 
- What is gradient descent ? 
	=> A technique to use the derivative of a cost function to change the parameter values, in order to minimize the cost 

## Gradient Descent 
- Main goal : change the parameter values so as to minimize the cost 
- Calculate the gradient of the cost function 
	=> The slope of the surface ate every point and the direction of the gradient is the direction of the greatest uphill 
- Gradient descent takes smaller steps toward the minimum function 
- derivative of J(Theta)

```
+-----------------------------------------+
| dJ/dTheta1 = -1/m*Sum[y^i - yhat^i)x1^i | ... for theta1  
+-----------------------------------------+
```

```
+-----------------------------------+
| NewTheta = OldTheta - Niu*DeltaJ |
+----------------------------------+
```


Where : 
- Niu : Learning rate (lenght of step until the minimum value)
- DeltaJ : Vector of the derivative of J(Theta1, Theta2, Theta3 ... )

## Training algorithm recap 
1. initialize the parameters Randomly 
	=> Theta^T = [Theta0, Theta1, Theta2, ...] 
2. Feed the cost function with training set, and calculate the error 
	=> J(Theta)
3. Calculate the gradient of cost function 
	=> DeltaJ = [dJ/dTheta1, dJ/dTheta2, dJ/dTheta3 ...]
4. Update weights with new values 
	=> NewTheta = OldTheta - Niu*DeltaJ
5. Go to step 2 until cost is small enough 
	=>
6. Predict the new customer X 
	=> P(y=1 | x) = Sigma(Theta^T X) 

# Lab: Logistic Regression

/// Support Vector Machine
# Support Vector Machine (SVM)
- Supervised algorithm that classified cases by finding a separator
1. Mapping data to a high-dimensional feature space 
2. Finding a separator (by hyperplane)

## Data transformation

``` 
- 1D => 2D => 3D ....
- using kernelling  : 
	=> Linear 
	=> Polynomial
	=> RBF 
	=> Sigmoid 
	...
+---------------------+
| Phi(x) = [x, x^2]	  |
+----------------------+
```


## Find hyperplane 
- SVM is absed on the idea of find a hyperplane that best devide the dataset int two classes:

```
 w^T x + b = -1
 w^T x + b = 0 
 w^T x + b = 1
```

 ## Pros and Cons 
 Pros : 
 - Accurate in high-dimmensional
 - Memory efficient 
Cons : 
- Prone to over-fitting
- No probability estimation 
- Small datasets (<1000 rows )

## Applications 
- Image recognition
- Text category assignement
- Detecting spam 
- Sentiment analysis 
- Gene Expression Classification 
- Regression, Outlier detection and clustering 

# Lab: SVM 

Quiz

## W4: Clustering

//// k-Means Clustering

# Intro to Clustering
- Dataset example : Customer data for segmentation study 
- finding clusters in a dataset unsupervised
- cluster : group of objects that are similar to other objects in the cluster, and dissimilar to data points in other clusters 

## Clustering vs Classification
- Classification algorithms predict categorical classed labels
	=> assigns instances to predefined classes such as defaulted/not defaulted 
	=> model/algo : kNN, Decision Tree, logistic regression ...
- Clustering : the data is unlabeled and the process is unsupervised

## Clustering applications 
- RETAIL/MARKETING : 
	=> Identifying buying patterns of customers (based on geographic location ...)
	=> Recommendation new books/movies to new customers  
- BANKING : 
	=> Fraud detection in credict card use 
	=> Identifying clusters of customers(e.g., loyal)
- INSURANCE : 
	=> Fraud detection in claims analysis 
	=> Insurance risk of customers 
- PUBLICATION : 
	=> Auto-categorizing news based on their content 
	=> Recommending similar news articles 
- MEDECINE : 
	=> Characterizing patient behavior 
- BIOLOGY : 
	=> Clustering genetic markers to identify family ties 
## Usage : 
- Exploratory data analysis
- Summary generation 
- Outlier detection 
- Finding duplicates 
- Pre-processing step 

## Clustering algorithms
- partitioned-based clustering : produces sphere-like clusters ... (medium large sized db)
	=> Relatively efficient 
	=> Ex : k-Means, k-Median, Fuzzy c-Means
- Hierarchical clustering : produces tree clusters ... (small sized data)
	=> Produces trees of clusters 
	=> Eg : Agglomerative, Divisive ... 
- Density-based Clustering 
	=> Produces arbitrary shaped clusters 
	=> Eg : DBSCAN 

# Intro to k-Means
- application : 
	=> customer segmentation 
- k-Means algorithms : 
	=> partitioning clustering 
	=> k-means divides the data into non-overlapping subsets(clusters) without any cluster-internal structure 
- Examples within a cluster are very similar 
- Examples across different clusters are very different 

- calculate similarity/dissimilarity
- k-means : intra-cluster distances are minimized and maximized
## calculate similarity : 
- Euclidean distance, cosine similarity, average distance and so on ... 

## k-Means algorithm
1. Initialize k=3, 
- centroids randomly 
	=> choose randomly 3 observations from the dataset and use these observations as the inital means 
	=> create 3 random point as centroids : C1[x1, y1], C1[x1, y1], C3[x3, y3] (for k=3) in the plan (2D)
2. Distance calculation 
- using matrix of distance (Distance Matrix)
- each row represents the distance of a customer from each centroid 

```
	C1			C2 		  C3
 [d(p1, c1) d(p1, c2) d(p1, c3)]
 [d(p2, c1) d(p2, c2) d(p2, c3)]
 [d(p3, c1) d(p3, c2) d(p3, c3)]
 [d(pn, c1) d(pn, c2) d(pn, c3)]
```

- Main objective of k-Means : is to minimize the distance of data points from the centroid of this cluster and maximize the distance from other cluster centroids. 

1. Assign each point to the closest centroid
- find the closest centroid to each data point
	=> how using DISTANCE MATRIX 

- SSE = the sum of the squared difference btw each point and its centroid

```
    +----------------------+
=>  | SSE = Sum(xi - Cj)^2 | ... where i = 1 ... n 
    +----------------------+
```


- To minimize the error : total distance of all members of a cluster from its centroid be minimized 
	=> move centroids!!!!

1. Compute the new centroid for each cluster 
- How ? 
	=> Change the centroid position
	=> each of the three cluster becomes the new mean
	=> recluster again and again  

1. Repeat until there are no more changes 
- each iteration moves the centroid 
- calculate the distance from new centroid and assign data points to the nearest centroid 
- The clusters with minimum error or the moste dense cluster 
Cons : no garantee to a global coverage 
=> rechoose the initial random centroid 
=> repeat the calculation again until the outcome be the optimum 

# More on k-Means
1. Randomly placing k centroids, one for each cluster 
2. Calculate the distance of each point from each centroid 
3. Assign each data point(object) to its closest centroid, creating a cluster 
4. Recalculate the position of the k centroids 
5. Repeat the steps 2-4, until the centroids no longer move

## k-Means accuracy 
- External approach
	=> Compare the clusters with the ground truth, if it is available
	=> we don't use this approach in real world problems because k-Means is unsupervised
- Internal approach 
	=> Average the distance btw data points within a cluster
## Choosing k 
- number of cluster in the dataset 
- it depends on the shape and scale of the distribution of points in a dataset 
	=> General approach : run the cluster with different value of k  and the matrix od accuracy for clustering
	=> Generally increse the k will always reduce the error
	=> The elbow point is determined where the rate of decrease sharply shifts.

## Recap 
- Med and Large sized databases (Relatively efficient)
- Produces sphere-like clusters 
- Needs number od cluster(k)

# Lab: k-Means

//// Hierarchical Clustering (HC)
# Intro to Hierarchical Clustering
- build a hierarchy of clusters where each node is a cluster consists of clusters of its daughter nodes
- two types : 
	=> Divisive (top-down) :  starts all observations in a large cluster and break it down into smaller pieces 
		=> Divisive : Dividing the clusters 
	=> Agglomerative (botton-top) : opposite to Divisive. Each observation start in its own cluster and pairs of clusters are merged together as they move up the hierarchy  
		=> Amass/collect things ...
		=> Most popular among Data Scientist 

## Agglomerative clustering
- builds the hierarchy from the individual elements by progressively merging clusters 
- visualize as Dendrogram 
	=> x: Each merge is represented by horizontal line 
	=> Y : two or more cluster merged (viewed as singleton clusters) 
	=> by moving from down-up, we can reconstruct the history of merges

# More on Hierarchical Clustering
1. Create n clusters, one for each data point 
2. Compute the proximity Matrix 
3. Repeat
	3.1 Merge the two closest clusters 
	3.2 Update the proximity matrix 
4 Until only a single cluster remains

## Distance btw clusters
- the distance of a single data point(one feature) can be calculated using the Similarity/dissimilarity matrix 
	=> however the distance btw two datapoint(more than two features) within the a cluster look more complicated
- it depends on the nature of dataset, size  ... 

### General creterial ####
- Single-Linkage Clustering 
	=> Minimum distance btw cluster
- Complete-Linkage Clustering 
	=> Maximum distance btw clusters
- Avg Linkage Clustering
	=> Avg distance btw clusters 
- Centroid Linkage Clustering 
	=> Distance btw cluster centroids 

## PROS VS CONS 
PROS : 
- Doesnt required numer of cluster to be specified 
- easy to implement 
- Produces a dendrogram, which helps with understanding the data 

CONS : 
- Can never undo any previous steps throughout the algorithm 
- Generally has long runtimes 
- Sometimes difficult to identify the number of clusters by the dendrogram 

## Hierarchical clustering Vs k-Means 
K-Means --------------------- Hierarchical clustering
1. Much more efficient ----- Can be slow for large datasets 
2. required the number of clusters to be specified ---- Doesnot require the number of clusters to run 
3. Gives only one partitioning of the data based ----- Gives more than one partitioning depending on the resolution 
on the predefined number of clusters 
4. Potentially returns different clusters each ------------- Always generates tha same clusters 
time it is run due to random initialization of centroids 

# Lab: Agglomerative clustering

//// Density-based Clustering
# DBSCAN
- used for spatial data analysis
- locates regions of high density, and separates outliers (or region of low density)
- density : numbers of points withing a specified radius 
- Most popular : DBSCAN !!!

## DBSCAN for class identification
- find out any arbitrary shaped cluster without getting effected by noise 
- how it works ? 
	DBSCAN (Density Based Spatial Clustering of Applications with Noise)
	- Is one of the most common clustering algorithms
	- Works based on density of objects  
	=> R(Radius of neighborhood)
		=> Radius(R) that if includes enough number of points within, we call it a dense area 
	=> M(Min number of neighbors)
		=> The minimum
identify the type of point : 
	=> core point : in the neighborhood are at least M points
	=> border point : neighborhood contains less than M points/it's reachable from some core point
	=> outlier point : not core point neither reachable from core point 
group the point as cluster 

## Advantages
1. arbitrarily shaped cluters 
2. Robust to outliers 
3. Does not require specification of the number of clusters 

# Lab: DBSCAN Clustering

## W5: Recommender Systems

# Intro to Recommender Systems
- Recommender systems capture the pattern of peoples behavior and use it to predict what else they might want or like. 
- people tend to like things in the same category/things that share the same characteristics 

## Applications :
- What to buy ? 
	=> E-commerce, books, movies, beer, shoes ...
	=> Amazon : 

- Netflix : everything is driven by customer selection
	=> if a movie gets viewed frequently or not.
	=> Its recommendation systems ensures that movie get an increasing number of recommendation
- The recommender engine is used to recommend anything from where to eat/ what job to applys to. 
- Where to eat ? 
	=> UberEats
- Which job to apply to ?
	=> Linkedin : 
- Who you should be friends with ? 
	=> Facebook, instagram, Twitter ...
- Personalize your experience on the web 
	=> News platforms, news personalization ...
	=> News platform website, a recommender system makes note of : types of stories that you clicked on/make recommendation on which types of stories you might be interested in reading in future ...
## Some advantages : 
- Broader exposure
- Possibility of continual usage or purchase of products 
- Provides better experience 

## Types recommender systems
1. Content-Based : Show me more of the same of what I've liked before(visually browsing...) => Similar items"
	=> User's favorite aspects of an items are, and then make recommendations on items that shares thoses aspects  
2. Collaborative Filtering : "Tell me what's popular among my neighbors, I also might like it => Similar preferences"
	=> based on trends of what other people like 
3. Hybrid recommendations systems : combines both 

## Implementing recommender systems
1. Memory-based 
	- Uses the entire user-item dataset to generate a recommendation 
	- Uses statistical techniques to approximate users or items 
		e.g : Pearson Correlation, Cosine Similarity, Euclidean Distance, etc 

2. Model-based 
	- Develops a model of users in an attempt to learn their preferences 
	- Models can be created using Machine Learning technoques like regression, clustering, classification, etc ...

# Content-based Recommender Systems : 
- Try to recommend items to users based on their profile 
- The user's profile revolves around that users preferences and tastes
- it's shaped based on : 
	=> user ratings 
	=> number of times that user has clicked on different items or perhaps even liked those items
- The recommendation process is based on the similarity between those items 
	=> similarity is measured based on the similarity in the content of those items 
- Contents means : category, tag, genre, ...

## How it works ? 
- Dataset shows movies that our user has watched and also the genre of each of the movies ...
- To make the recommendation based on the movies watched by the users : 
1. Create user profile : 
	- Weighing the genres 
	- User's rating for the movies that he's already watched 
		=> mv1 : 2
		=> mv2 : 10
		=> mv3 : 8  ... where mv : movies watched 
			=> Input User Ratings (vector )
	- encode the movies through the one-hot encoding approach 
		=> create a feature set matrix with 3 movies (rows) => dim(3,1)
		=> genre : comedy, adventure, super hero, scifi (col) => dim(4,4)

```
+---------------------------------------------------------------------+
| Input User Ratings (vector ) x Movie Matrix = Weighted Genre Matrix | => dim(4, 4)
+---------------------------------------------------------------------+
```

=> aggragate the user profile => new matrix dim (1, 4)
=> Normalization to refind what's the use likes to what the most 

1. Finding the recomendation : 
=> Movie Matrix <=> new weighted movies matrix => Weight average matrix (for final recommendation...)

# Collaborative Filtering
- Relationship exists btw product and people's interests 
- two types of Collaborative Filtering
1. User-based collaborative filtering
2. Item-based collaborative filtering 


1. User-based collaborative filtering
	- Based on on users similarity/neighbordhood 
	- Collaborative Filtering engine first looks for users who are similar 
		=> users who share the active users rating patterns : 
		=> these similarities are based on things like : history, preference, and choices that users make when enjoying, watching, or enjoying something ...
		=> uses the rating from these similar users to PREDICT the possible ratings by the active user for a movie that she had NOT previously watched 

Ex : User ratings matrix : 
- 4 users (u1 ...n)
- 5 different movies (mv1 ... m => movies categories )
  
``` 
			mv1  mv2 mv3 mv4 mv5
			+---+---+---+---+---+
		u4	|9	|6	|8	|4	|	|
			+---+---+---+---+---+
		u3	|2	|10	|6	|	|8	|
			+---+---+---+---+---+
		u2	|5	|9	|	|10	|7	|
			+---+---+---+---+---+
Active =>u1	| ?	|10	|7	|8	|?	|x
user  		+-------------------+
			Rating Matrix
```

=> where : 
	=> u1 : Active users
	=> ?  : Movies which the active user hasn't watched yet
	=> Goal : Recommend the unwatched movies to the user !!!
	1. Calculate the similarity of active among other users (who have already watched the movies)
		- Tools : Distance/similarity measurements : Euclidean Distance, Pearson Correlation, Cosine Similarity...
		- To calculate the level of similarity btw 2 users: 
			=> look at the movies rated by all the users in the past 
			=> ex : u1 vs u2 = 0.7 | u2 vs u3 = 0.9 | u3 vs u4 =0.4
	2. Create a Weighted rating Matrix : user's neighbors opinion based on the numbers of candidate movies for recommendation 
		- Ratings Matrix Subset x Similarity Matrix = Weighted ratings matrix  
		- This gives more weight to the ratings of those users who are more similar to the active user 
	=> Based on items similarity
- Recommendation matrix based on all of the weighted rated:
  
```
		     mvx| mvy| weight sum
			+---+----+-----------+
u2 + u3		|	|12.1|0.9+0.7
			+---+----+-----------+
(u2+u3+u4)	|8.9|	 |0.4+0.9+0.7|
			+---+----+-----------+ 
```

- sum of similarity Index btw users (u'2 + u'3) then (u'2 + u'3+u'4= 
- sum of weighted rating (u2 + u3) then u2 + u3+u4 then divided by the their respective similarity index for normaization 

```
	mx         	 				my
			 			 (u2 + u3)/(u'2 + u'3)
(u2+u3+u4)/(u'2+u'3+u'4)

Normalized recommendation matrix 
		     mv1| mv2 
			+---+----+
u1(target)	|4.4|7.5
			+---+----+
```

- The result is the potential rating that our active user will give to theses movies based on her similarity to other users 

## Collaborative Filtering (User based vs Items-based ) : 

- User based  :
	=> the recommendation is based on users of the same nighborhood with whom he or she shares common preferences
	=> if a u1 and u2 both liked the same items, they considered as similar/neighbor user 
	=> we can recommend these items to another potential users based on positive rating by u1 and u3 

- Items-based :
	=> similar items build neighborhoods on behavior of users 
	=> However that is not based on their contents 
	=> two items are considered as similar/neighbor if they are liked by two different users

## Challenges of collaborative filtering

- Data sparsity : 
	=> Users in general rate only a limited number of items 
- Cold start :  
	=> Difficulty in recommendation to new users or new items 
- Scalability : 
	=> Increase in number of users or items

/!\ Solution : Hybrid based recommender systems 

# Lab: Content-based Recommendation Systems
# Lab: Collaborative Filtering on Movies

Final Quiz 

## W6: Final Project


## References



