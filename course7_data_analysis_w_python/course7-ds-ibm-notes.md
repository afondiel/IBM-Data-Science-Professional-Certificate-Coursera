# Course 7: Data Analysis with Python 

**What you'll learn:**

- Develop Python code for cleaning and preparing data for analysis - including handling missing values, formatting, normalizing, and binning data
- Perform exploratory data analysis and apply analytical techniques to real-word datasets using libraries such as Pandas, Numpy and Scipy
- Manipulate data using dataframes, summarize data, understand data distribution, perform correlation and create data pipelines
- Build and evaluate regression models using machine learning scikit-learn library and use them for prediction and decision making

**Skills you'll gain:**, Machine Learning, regression, Hierarchical Clustering, classification, SciPy and scikit-learn

## Table of Contents

- [W1: Importing Databases](#w1-importing-databases)
- [W2: Data Wrangling/Munging](#w2-data-wranglingmunging)
- [W3: Exploratory Data Analysis (EDA)](#w3-exploratory-data-analysis-eda)
- [W4: MODEL DEVELOPMENT](#w4-model-development)
- [W5: MODEL EVALUATION AND REFINEMENT](#w5-model-evaluation-and-refinement)
- [W6: Final Exam](#w6-final-exam)
- [References](#references)


## W1: Importing Databases

# The Problem
## Why Data Analysis ? 
- Data is everywhere
- Data analysis/data science help us answer questions from data
- Data analysis plays an important role in : 
	=> Discovering useful information
	=> Answering questions
	=> Predicting future or the unknown

## Define a Data Analysis problem ? 
- Problem : Estimate used car prices
	=> Is there data on the prices of other cars their characteristics ? 
	=> What features of cars affect their prices ? 
		-> Color? Brand ? Horsepower ? Something else ?
	=> Asking the right questions in terms of data 

# Understanding the Data
- source data (1985) : 
	=> dataset automobile : 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos'
- data available is row data/no proper/ untructure data 
	=> Need to be clean and prepare ( Header, delete empty data ...)
	=> The description the data : data attributes (names, range ... of each column)

- input attributes (x : independent var) => features ... / Predictors
- output attribute (y : dependent var) => Target/label/ value to be predicted

# Python Packages for Data Science

## Group I : Scientif Computing Libraries 
- Pandas : Data structures & tools
- NumPy : Array & Matrices
- SciPy : integrals, solving differential eq, optimization

## Group II : Visualization Lib
- Matplotlib : plots & graphs, most popular
- seaborn  : plots (heat maps, time series, violin plots ..) 

## Group III : Algorithms Lib
- Scikit-learn : Machine Learning  : regression, classification ...)
- Statsmodels : Explore data, estimate statistical models, and perform statistical tests...) 

# Importing and Exporting Data in Python
- process of loading and reading data into Python from various resources 
- Two main propreties : 
	=> Format : .cvs, .json, .xlsx, .hdf ... 
	=> File Path of dataset  : 
		=>computer (/Desktop/ maydat.csv)
		=>internet : https://archive.ics.uci.edu/ml/machine-learning-databases/autos'
- getting Data : using 'pandas' lib  
- printing data to make sure everything ok  : df.head(n), df.tail(n) => n : nb of rows
	=> replace a default row : df.column = ['a', 'b' ....]


- exporting different data format in python w/ pandas 
=> read : pd.read_formatdata() 
=> save : df.to_formatdata()
where => formatdata: csv, json, excel ...

# Getting Started Analyzing Data in Python
- Understand the data before you begin any analysis
	=> Things to check : Data types, Data distribution
	=> Locate potential issues w/ the data
- df.dtypes : to check if data type make are coherent
- df.describe()/describe(include ="all") : statistical summary/distribution 
- df.info() : shows the botton 23 rows and top 23 rows

# Accessing Databases with Python
- see sql database course => course 6

# Lab 
- dataset : https://archive.ics.uci.edu/ml/datasets/Automobile?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2021-01-01

# Quiz

**Summary:** 

- Define the Business Problem: Look at the data and make some high-level decision on what kind of analysis should be done
- Import and Export Data in Python: How to import data from multiple data sources using the Pandas library and how to export files into different formats.
- Analyze Data in Python: How to do some introductory analysis in Python using functions like dataframe.head() to view the first few lines of the dataset, dataframe.info() to view the column names and data types.

## W2: Data Wrangling/Munging

# Pre-processing Data in Python (Data Cleaning/ Data wrangling)

- Process of converting/mapping data from the initial "raw" form into another format
	=>preparing the data for data analysis 
- Operation are done along the columns of the DataFrame.
- Each column is a Pandas series 
	=> addiction of 1 for each row of the col a  : df['a'] = df ['a'] + 1  


- occurs when no data value is tored for a variable (feture) in an observation
- Missing values can be : 0, ?, N/A, empty/blank cell ...
## Dealing w/ missing data ?
- Check w/ the data collection source
- Drop the missing values
	=> drop the variable
	=> drop the data Entry 
- Replace the missing values
	=> replace it with an average (of similar datapoints) => to avoid waste
	=> replace it by the frequency (data that appears more often) 
	=> replace it based on other functions 
- Leave it as missing data 

## Drop missing values 
- df.dropna() => df.drona(subset=["price"], axis=0, inplace=True) : inplace to be taken immediatly
- df.replace(df['a'], missing values, new)
- high quality data or source 
- ## Replace data with mean (for numerical values)
/*mean = df['a'].mean()
df.replace(df['a'], np.nan, mean)*/

# Data Formatting in Python
- data are collected from different places and stored in different formats
- bringing data into common stardand of expression allows users to make meaning comparison 
- abreviation, synonyms, local words, data types => convert to a classical and understandble format. 
- clean data for strong model 
- check data type if not ok => convert to the wanted format

# Data Normalization in Python
- Uniforms the features value with different ranges
- to make easier comparison btw two or more values(correlation ?) 
- similar intrisec influence on analytic model  
## Methods : 
- Simple Feature scaling : Xnew = Xold/Xmax => range[0:1]
- Min-Max : Xnew = Xold = (Xold - Xmin)/(Xmax - Xmin)) => range[0:1]
- Z-score : Xnew = (Xold - µ ) / sigma => range[0:0]

# Binning in Python
- Grouping of values into "bins"
- Converts numeric into categorical variables
- group a set of numerical values into a set of bins =>[low, medium, high]
- value great categorization 

# Turning categorical variables into quantitative variables in Python
- using dummies methods

# Data Wrangling
- Data wrangling is the process of converting data from the initial 
format to a format that may be better for analysis.

### Lab 
### Quiz

## W3: Exploratory Data Analysis (EDA)

# Exploratory Data Analysis


- Preliminary step in data analysis to
	=> Summarize main characteristics of the data
	=> Gain better understand of the dataset
	=> Uncover relationships between variables
	=> Extract import variables

Question : 
	What are the characteristics which have the most impact on the car?  

# Descriptive Statistics (course 3)
- Describe basic features of data 
- Giving short summaries about the sample and measures of the data 
- Help to describe basic features of a dataset and obtain a short summary about the sample and measures of the data 

## Descriptive statistics functions :
- pandas df.describe() : Summarize statistics
- pandas Value_Counts() : Summarize the categorical data
- seaborn sns.boxplots() : visualize numeric data (various distributions)
	=> Upper Extreme 
	=> Upper Quartile
	=> Median
	=> Lower Quartille
	=> Whisker
	=> Lower Extreme
- scatter plot() : to visualize continue data (bunches of points) 
	=> Each observation represent as a point
	=> Scatter plot show the relationship between two variables
	1. Predictor/independent variables on x-axis
	2. Target/dependent variables on y-axis

# GroupBy in Python
- Relationship btw two or more variables (df['a'], df['b'] ...) 
- Can be applied on categorical variables 
- Group data into Categories
- Single or multiple variables 
## pandas Pivot() => One variable displayed along the columns and the other variable displayed along the rows 
## Heatmap  : Plot target variable over multiple variables

# Correlation
- Measures to what extent different variables are interdependent
ex : Smoking => Lung Cancer 
ex : Rain => Umbrella
- Correlation doesn't imply causation 
types of correlation : 
	=> positive correlation
	=> negative correlation 
	=> weak correlation

# Correlation - Statistics
## Pearson Correlation : 
- Measure the strenght of the correlation between two features
	=> Correlation coefficient
	=> P-value

- Correlation coefficient (r)
	=> close to +1: Large Positive relationship 
	=> close to -1: Large Negative relationship 
	=> close to 0: No relationship 
- P-value 
	=> P-value < 0.001 : Strong certainty in the result
	=> P-value < 0.05 : Moderate certainty in the result
	=> P-value < 0.1 : Weak certainty in the result 
	=> P-value > 0.1 : No certainty in the result

- Strong Correlation : 
	=> Correlation coefficient close to 1 or -1 (range [-1,1]
	=> P value less than 0.001 

# Association between two categorical variables: Chi-Square(X^2)

## Categorical variables 
- We use Chi-square Test for Association (denoted as X^2)
- The test is intended to test how likely it is that an observed distribution is due to chance
## Chi-square Test for association 
- The Chi-square tests a null hypothesis that the variables are independent
- The Chi-square does not tell you the type relationship that exists btw both variables; but only that a relationship exists

```
------------------------------
|| X^2 = Sum (Oi - Ei)^2/Ei ||
------------------------------
 - Oi : observed values
 - Ei : Expected values

=> To get expected values(From observed values/table) : 

   Row total*Column total 
	-----------------------
		 Grand total 

------------------------------------------------
|| Defree of freedom = (row - 1)*(column - 1) ||
------------------------------------------------
```

- deviation == difference (btw two values)

# Summary  : 
Describe Exploratory Data Analysis: By summarizing the main characteristics of the data and extracting valuable insights.
Compute basic descriptive statistics: Calculate the mean, median, and mode using python and use it as a basis in understanding the distribution of the data.
Create data groups: How and why you put continuous data in groups and how to visualize them.
Define correlation as the linear association between two numerical variables: Use Pearson correlation as a measure of the correlation between two continuous variables
Define the association between two categorical variables: Understand how to find the association of two variables using the Chi-square test for association and how to interpret them.

# Lab
# Grade
# Reference : 
Exploratory data analysis : https://en.wikipedia.org/wiki/Exploratory_data_analysis
pandas functions : https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm

## W4: MODEL DEVELOPMENT

# Model Development
- Simple and multiple Linear Regression 
- Model Evaluation using Visualization
- Polynomial Regression and Pipelines
- R-squared and MSE for In-Sample Evaluation
- Prediction and Decision Making 

/!\ Question: How can you determine a fair value for a used car ?

## Model Development 
- A model can be thought of as a mathematical equation used to predict a value given one or more other values
- Relation one or more independent variables(features) to dependent variables(target) 
|features| => |Model| => |target|
- Usually the more *relevant data* you have the more accurate your model is

# Linear Regression and Multiple Linear Regression
- Linear regression will refer to one independent variable to make a prediction
- Multiple regression will refer to multiple independent variables to make a prediction 
## Simple Linear Regression
- The predictor(independent) variable: x 
- The target (dependent) variable: y 

```

---------------
|y = bo + b1*x|
---------------
```

where => bo : the intercept 
		 b1 : the slope 

## Simple Linear Regression : Prediction 
- Chose some "hypothetical" inputs value that best match w/ the regression line to fit the model
## Simple Linear Regression : Fit 
- training the model 
- Predictor and target are storage in two Numpy Arrays X[predictors] = Y[Target]
- Noise : random value added to the model to increase the accuracy 

```
--------------- 	--------- 		--------
|Input values | => |	Fit	 | => | Predict | (y^ = bo+b1*x) => estimated model
---------------		----------		---------
		||<============================||
			Retrain the estimatedd values
```

- To fit the model we use *scikit-learn* library 
1. Import linear model
2 Create Linear Regression Object Using the constructor
=> We define the predictor variable (X = df[['highway-mpg']]) and target variable(Y=[['price']])
=> Then : lm.fit(X, Y) also parameters bo, b1
=> Predict  : lm.predict(X)

SLR - Estimated Linear Model 
=> bo, b1 ...

## MLR - Estimated Linear Model 

```
---------------------------------------
|y = bo + b1*x1 + b2*x2 + b3*x3 + ...+ |
----------------------------------------
```
- Same step as the SLM !!!!

# Model Evaluation using Visualization
## Why use Regression plot ? 
- The relationship btw 2 variables
- the strength of the correlation
- The direction of the relationship (positive or negative) 
- Combination of : 
	=> The scatterplot : where each point represebts a different Y	
	=> The fitted linear regression line (y)

- sns.Regplot() => to viz the regression
## Residual plot 
- Randomly spread out around x-axis which means the linear model is appropriate
- Expected behaiviour : results have mean btw the same VARIANCE
- no curvature (values changes w/ axis) => FOR LINEAR REGRESSION!!!!
- sns.residplot() => to display plot

- predicted values  are discret values => pandas converted into distribution (Gaussian)

## Distrubution plots  : 
- The fitted values that result from the model
- The actual values 

Predicted values shall be much closer to the TARGET VALUES!!!! 

# Polynomial Regression and Pipelines
- Used when linear model is not the best fit for our data 
- A special case of the general linear regression model 
- Useful for describing curvilinear relationships
	=> Curvilineat relationships
	=> By squaring or setting higher-order terms of the predictor variables

- Quadratic - 2nd order
```
------------------------------
|Y^ = bo + b1*x1 + b2*(x1)^2 | (square model)
------------------------------
```

- Cubic - 3rd order
```
-----------------------------------------
|Y^ = bo + b1*x1 + b2*(x1)^2 + b3(x1)^3  | (Cubic model)
-----------------------------------------
```

- Higher order (when a good fit hasn't been achived by the lower order)
```
------------------------------------------------
|Y^ = bo + b1*x1 + b2*(x1)^2 + b3(x1)^3 +...+  | (Cubic model)
-----------------------------------------------
```

- The relationship btw the variables and parameters is always linear 
- function to train the model  : np.polyfit(x, y, n) => n : order
- for more complex polynom better user scikit-learn lib

## pipelines

- way to simply the code
```
---------------- 	---------------------	  -------------------
|Normalization | => |Polynomial transform| => |Linear Regression |
----------------    ----------------------	  --------------------
```
- pipeline lib => Create pipeline Object (pipe = Pipe())
- Pipe.fit()
- Pipe.predict()

# Measures for In-Sample Evaluation
- A way to numeriaclly determine how good the model fits on dataset 
- Two import measures to determine the fit of the a model
	=> Mean Square Error (MSE) 
	=> R-squared (R^2)
## Mean Square Error
- difference(deviation) btw the actual value y and the predicted value Y^ squared
- add every error(difference) and divide by the number of the samples
- to define the function we can import it from  : sklearn.metrics lib (y, y^)

```
-------------------------
| MSE = 1/N*Sum(Y-Y^)^2 | 
--------------------------
```
=> Derived(MSE) = Gradient Descent !!!

## R-squared (R^2)
- The coef of determination or R squared(R^2)
- Is a measure to determine how close the data is to the fitted regression line
- R^2 : the percentage of variation of the target variable (Y) that is explained by the linear model 
- Think about as comparing a regression model to a simple model i.e the mean of the data points 
- If the variable x is a good predictor our model should perform much better than the mean 

```
-----------------------------------------------------------------
| R^2 = (1 - MSE of regression line/MSE of the avg of the data ) |
-----------------------------------------------------------------
```

=> MSE of regression line > MSE of the avg of the data
=> (MSE of the avg of the data) != 0  
=> R^2 ~ [0, 1] => the line is a good fit fo the data
	=> lm.fit(X, y) 
	=> lm.score(X, y) 

# Prediction and Decision Making
- How can we measure that model is corrected  ?
## Determining a Good Model Fit
- To determine final best fit, we look at a combination of :
=> Do the predicted values make sense 
=> Visualization 
=> Numerical measures for evalueation
=> Comparing Models

**Summary:** 

```
- Define the explanatory variable and the response variable: Define the response variable (y) as the focus of the experiment and the explanatory variable (x) as a variable used to explain the change of the response variable. Understand the differences between Simple Linear Regression because it concerns the study of only one explanatory variable and Multiple Linear Regression because it concerns the study of two or more explanatory variables.
- Evaluate the model using Visualization: By visually representing the errors of a variable using scatterplots and interpreting the results of the model.
Identify alternative regression approaches: Use a Polynomial Regression when the Linear regression does not capture the curvilinear relationship between variables and how to pick the optimal order to use in a model.
- Interpret the R-square and the Mean Square Error: Interpret R-square (x 100) as the percentage of the variation in the response variable y  that is explained by the variation in explanatory variable(s) x. The Mean Squared Error tells you how close a regression line is to a set of points. It does this by taking the average distances from the actual points to the predicted points and squaring them.
```

### Lab 
### Quiz 

## W5: MODEL EVALUATION AND REFINEMENT

# Model Evaluation and Refinement
- Tell how the model performs in the real world 
- In-sample evaluation tell how well the model will fit the data used to train it
	=> Problem ?
		> It does not tell how the trained model can be used to predict new data
	=> Solution ? 
		> In-sample data or training set 
		> Out-of-sample evaluation or test set
- Split the dataset into : 
	=> Training dataset (70%)
	=> Test dataset (30%)
- Build and train the model w/ a training set
- Use testing set to assess the performance of a predictive model
- When we have completed testing the model we should use all the data to train the model to get the best performance

- Function for splitting sets  : train_test_split() from scikit pkg

## Cross validation
- Most common out-of-sample evaluations metrics
- More effective use data (each observation is used for both training and testing)
- We split dataset into pieces either for training and testing (70%/30% and vice-versa)
	=> Then we take the average results as the estimate of out-of-sample error 
- the validation result depend on the type of MODEL
- To apply cross validation => cross_val_score() from scikit-learn pkg 

- cross_val_predict()
- It returns the prediction that was obtained for each element when it was in the test set 
- Has a similar interface to cross_val_score() 


# Overfitting, Underfitting and Model Selection
- Study case : Polynomial Model n-order
-  Overfitting : Model too complex(order higher ) VS Less dataset 

-  Underfitting :To many errors => model too simple to fit the datan 
		=> Solution  : Increase the oder of the polynomial order 
		=> Too much data VS simple model

- Test error (parabola curve) > Training Error(negative exponentiel curve) 
- Optimal capacity  : (center)point btw under and over fitting.
- The difference btw test error and training error = generalisation gap 

- irreducible error == Noise (can't be reduced)

# Ridge Regression Introduction
# Ridge Regression
- Just like Polynomial reg but w/ a Parametrable coef called Alpha
- Rigle() is imported from sklearn.linear_model 

/!\ The model that exhibits  overfitting is usually the model with the lowest parameter value for alpha

# Grid Search
- Allows to scan through multiple free parameters w/ few lines of code
- Alpha == HYPERPARAMETER
- skilearn => Grid Search : iterating over HYPERPARAMETER using cross-validation 

- split data into 3 sets : 
 => Training (MSE / R^2)
 => Validation (HYPERPARAMETER)
 => Test (MSE / R^2)

 ## Grid Search CV
 - takes scoring methods,nb of folds(R^2), the model and free parameters ...   

**Lesson Summary**

- Identify over-fitting and under-fitting in a predictive model: Overfitting occurs when a function is too closely fit to the training data points and captures the noise of the data. Underfitting refers to a model that can't model the training data or capture the trend of the data.
- Apply Ridge Regression to linear regression models: Ridge regression is a regression that is employed in a Multiple regression model when Multicollinearity occurs.
- Tune hyper-parameters of an estimator using Grid search: Grid search is a time-efficient tuning technique that exhaustively computes the optimum values of hyperparameters performed on specific parameter values of estimators.

### Lab 
### Quiz

## W6: Final Exam 


## References



