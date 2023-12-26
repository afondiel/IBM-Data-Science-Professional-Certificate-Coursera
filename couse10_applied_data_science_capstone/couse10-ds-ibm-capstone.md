# Course 10: Applied Data Science Capstone

![Spacex Falcon Heavy dual landing](https://media4.giphy.com/media/3ohs4gSs3V0Q7qOtKU/giphy.gif)

**What you'll learn**

- Demonstrate proficiency in data science and machine learning techniques using a real-world data set and prepare a report for stakeholders   
- Apply your skills to perform data collection, data wrangling, exploratory data analysis, data visualization model development, and model evaluation
- Write Python code to create machine learning models including support vector machines, decision tree classifiers, and k-nearest neighbors  
- Evaluate the results of machine learning models for predictive analysis, compare their strengths and weaknesses and identify the optimal model   

**Skills you'll gain:** Data Science, Data Analysis, CRISP-DM, Methodology, Data Mining

## W1: Project Introduction

**Objectives:**

- Use data science methodologies to define and formulate a real-world business problem.
- Use your data analysis tools to load a dataset, clean it, and find out interesting `insights` from it

### Problem/Issue to solve: 

- In this capstone, we will predict if the Falcon 9 first stage will LAND successfully. 
- SpaceX advertises Falcon 9 rocket launches on its website, with a cost of 62 million dollars; 
other providers cost upward of 165 million dollars each, much of the savings is because SpaceX can reuse the first stage. 
- Therefore if we can determine if the first stage will land, we can determine the cost of a launch. 
- This information can be used if an alternate company wants to bid against SpaceX for a rocket launch. 

### Solution Steps:

1. Course Introduction & Understanding: 

```
# Project Scenario and Overview
# Obtain an IBM Cloud Feature Code : Ungraded External Tool
# IBM Watson Account creation and Watson Studio
# Getting started with GitHub
# Publishing Notebook to Github
# IBM Watson Setup and Project Creation
# Adding a Notebook to the Project
```

2. Data Collection Overview:

```
# Lab: Complete the Data Collection API Lab
# Data Collection with Web Scraping
# Lab: Complete the Data Collection with Web Scraping lab 
# Quiz
```

3. Data Wrangling Overview

EDA & Training Labels:
- find patterns in the data
- datermine the LABEL for training supervised models or the OUTCOME

```
- True Ocean          => means the mission outcome was successfully landed to a specific region of the ocean while 
- False Ocean         => means the mission outcome was unsuccessfully landed to a specific region of the ocean. 
- True RTLS           => means the mission outcome was successfully landed to a ground pad 
- False RTLS          => means the mission outcome was unsuccessfully landed to a ground pad.
- True ASDS           => means the mission outcome was successfully landed on a drone ship 
- False ASDS          => means the mission outcome was unsuccessfully landed on a drone ship
- None ASDS/None None => these represent a failure to land
```

### Lab: Data Wrangling
### Quiz

## W2: EXploratory Data Analysis (EDA)

- In this module, you will collect data on the Falcon 9 first-stage landings. 
- You will use a RESTful API  and web scraping. 
- You will also convert the data into a dataframe and then perform some data wrangling.

Exploratory Data Analysis Overview:

```
Hands-on Lab: Complete the EDA with SQL
- (Optional)Hands-on Lab: Complete the EDA with SQL
- Ungraded External Tool•. Duration: 1 hour1h
- Check Points: Exploratory Analysis Using SQL
- Practice Quiz•3 questions
- Exploratory Data Analysis using SQL
- Quiz•5 questions
- •Grade: --

- Hands on Lab: Complete the EDA with Visualization lab
- (Optional)EDA with Visualization Lab
- Ungraded External Tool•. Duration: 1 hour1h
- Check Points: Complete the EDA with Visualization
- Practice Quiz•3 questions
- Exploratory Data Analysis for Data Visualization
- Quiz•3 questions
- •Grade:
```
### Lab: Data Wrangling
### Quiz


## W3: Interactive and Visual Analytics and Dashboard

- In this module, you will build a dashboard to analyze launch records interactively with Plotly Dash. 
- You will then build an interactive map to analyze the launch site proximity with Folium.

- Interactive Visual Analytics and Dashboards: 

```
- Hands on Lab: Complete the Data Visualization with Folium
- (Optional)Hands-on Lab: Interactive Visual Analytics with Folium lab
- Ungraded External Tool•. Duration: 1 hour1h
- Hands-on Lab: Build an Interactivce Dashboard with Ploty Dash
- Ungraded External Tool•. Duration: 1 hour1h
- Check Points: Interactive Visual Analytics and Dashboard
- Practice Quiz•7 questions
- Graded Quiz: Interactive Visual Analytics and Dashboard
- Quiz•5 questions
```

### Lab: Data Wrangling
### Quiz

## W4: Predictive Analysis (Classification)

- In this module, you will use machine learning to determine if the first stage of Falcon 9 will land successfully.
- You will split your data into *training data* and *test data* to find the best *Hyperparameter* for:
  - SVM
  - Classification Trees
  - Logistic Regression
  - Then find the method that performs best using test data.

Objectives:

- Split the data into training testing data.
- Train different classification models.
- Hyperparameter grid search.
- Use your machine learning skills to build a predictive model to help a business function more efficiently.

- Predictive Analysis Overview:

```
- Hands on Lab: Complete the Machine Learning Prediction lab
- (Optional)Hands-on Lab: Complete the Machine Learning Prediction lab
- Ungraded External Tool
- Check Points: Predictive Analysis
- Practice Quiz•4 questions
- Graded Quiz: Predictive Analysisis
```

### Lab: Data Wrangling
### Quiz

## W5: Present Your Data Driven Insights

- Submission Overview and Instructions

### Lab: Data Wrangling
### Quiz
