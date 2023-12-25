# Course 2: Tools for Data Science 

**What you'll learn:**

- Describe the Data Scientist’s tool kit which includes: Libraries & Packages, Data sets, Machine learning models, and Big Data tools   
- Utilize languages commonly used by data scientists like Python, R, and SQL   
- Demonstrate working knowledge of tools such as Jupyter notebooks and RStudio and utilize their various features    
- Create and manage source code for data science using Git repositories and GitHub.   

**Skills you'll gain** Data Science, Python Programming, Github, Rstudio, Jupyter notebooks

## Table of Contents

- [**W1 : Languages of Data Science**](#w1--languages-of-data-science)
- [**W2 : Open Source Tools**](#w2--open-source-tools)
- [**W3 : Watson Studio**](#w3--watson-studio)
- [**W4 : Create and Share Your Jupyter Notebook**](#w4--create-and-share-your-jupyter-notebook)
- [References](#references)

## **W1 : Languages of Data Science**

### Introduction

- Data is central to Data Science (data management, extraction, tranformation, analysis, visualization...)
- Data Science requires programming languages & frameworks
- Automation w/ Data Science tooling  : *save time, uncovered inspiration*
- Visual programming & modeling
- Open Sources(High TCO*) vs Commercial tools (Low TCO)
- Cloud Computing (used to speed up and facilitate DS work)

**TCO**: *Total Costs of Ownership*

### The languages of Data Science

`Kaggle 2019 survey` : learning new language plus 1Ok of wage increase. 

- Recommended : Python, R, SQL
- Optional : Scala, Java, C++, julia
- Others : JS, PHP, Go, Ruby, Visual Basics
Usage : Depend on the **problem** to be solved / or the **company** you in.

#### Roles in Data Science 
- Business Analyst
- Database Engineer
- Data Analyst
- Data Engineer
- Data Scientist
- Research Scientist
- Sw Engineer
- Statistician
- Product Manager 
- Project Manager

### Introduction to Python 
- Most popular DS language
- easy to learn
- great community 
- 80% DS use Python 
- Usage : AI, ML, WEB, IoT
- Organization : IBM, Wikepedia, Google, Yahoo, cern, FB ..

=> `What's python great for ?` 
- General Purpose language
- Large standard library 

=> For Data Science : 
- Scientific computing lib : Panda, NumPy, SciPy and Matplotlib
- For AI : PyTorch, TensorFlow, Keras, and Scikit-learn
- Natural Language Processing (NLP) : Natural Language TooKit (NLTK)

=> Diversity and Inclusion Efforts : 
Code of Conduct : https://www.python.org/psf/conduct/

### Introduction to R 

Open Source (OSI) vs Free SW (SFS) ? (=>different License )
- OSI : Business Focused
- FSF : Values Focused

- `R is a FSF ! `

=> Who's R For ? 
- Statistician, mathematicians, data miners (for statistical sw, graphing and data analysis )
- Array-oriented synthax : easy conversion 'maths => Code'
- ease to learn
- Organizations love it : IBM, Google, FB, TechCrunch, UBER, Trulia.. 

=> What R makes great ? 
- world largest repository of Statistical knowledge 
- More than 15000 release packages truns complex compository analysis
- R integrates well w/ C++, PYTHON, Java ..
- R is OOP 
- common maths operation (matrix are ease to do)

=> Large Community
- useR : user2020.r-project.com
- WhyR : www.whyr.foundation.com
- SataRdays : www.satarday.com
- Rladies : www.rladies.com

### Introduction to SQL

- SQL : Structured Query Language
- No procedure language
- 20y older than Python and R (creted on 1974 by IBM)
- useful to handle structured data 
  
=> `Relational DB` (collection of two dimmensional tables like row, column) : 
- ex : datasets, Microsoft Excel spread sheet  

=> `SQL elements` : 
- clause
- expression
- statement
- predicate
- query
 
=> What makes SQL Great : 
- job opportunnities in data engineering
- mandatory for Data Analyst and Data Science
- direct access to data when performing sql op (w/out copying the data in beforehand)
- interpreter between the user and the DB
- ANSI standard (scable : one db and multiple dbs)

=> SQL Databases  : 
- MySQM
- PostgreSQL
- Access
- SQLite
- Oracle DB
- MariaDB
- Microsoft SQL Server

### Others Data Science Language
- Scala : Apache Spark 
- Java : Weka (data mining), Java-ML(ml lib), Apache MLlib, Deeplearning4j
- C++ : TensoFlow, MongoDB, Caffe (Deeplearning)
- Julia : (create at MIT) Numerical analysis; complied language
- JS : Java Script : TensorFlow.js , R-js (Algebra)
- PHP, Go, Ruby, Visual Basics

### Categories of Data Science Tools

### Introduction 

```mermaid
graph LR;
  A[Data Management]-->B[Data Transformation Integration];
  B-->C[Data Visualization];
```

- [Data Management](#data-management) : is the process of persisting and retrieving data
- [Data Integration and Transformation](#data-integration-and-transformation) : (ETL : Extract - Transform - Load) is the process of retrieving data from remote data management systems
- [Data visualization](#data-visualization) : data Exploration process, final deliverable
- [Model building](#model-building) : create a ml or dl models using appropriate algo w/ lots of data
- [Model Deployment](#model-deployment) : makes ml/dl models accessible to a 3rd part application
- [Model Monitoring and Assessment](#model-monitoring-and-assessment) : continious perfomance qlity checks of deployed models (accuracy,fairness, adversial robustness )  
- [Code Asset Management](#code-asset-management) : visionning, collaborative features to facilite teamwork 
- [Data Asset Management](#data-asset-management) : dublication, backups, acess right management
- [Development Environment](#development-environment) : tools to implement, execute, test, and deploy
- [Execution Environment](#execution-environment) : where data processing, model training, Deployment take places
- [Fully integrated Visual Tools](#fully-integrated-visual-tools) : `integrates all`


### Open Source Tools for Data Science - Part 1 

#### Data Management tools
 
Relational tool : 
- MySQL 
- PostgreSQL 
NoSQL : 
- MongoDB
- Cassandra
- CouchDB 
File based:
- Hadoop File System
- Cloud File systems : Ceph
- Elasticsearch : text data storage & Data retrievement

#### Data Integration and transformation (ETL) tools 

- Extraction - Transform - Load or ELT because Data is dumped somewhere (i.e : Comes from some sources...)
- Data Scientist is responsible for Data
- Also called : Data Refinery and cleansing 
##### Tools 
- Apache Airflow :  created by RBNB
- Kubeflow : high level datascience pipeline layer on the top of Kubernetes
- Apache Kafka : created by Linkedin
- Apache NIFI : data viz editor
- Apache Spark SQL : scalable sql for cluster of multi-nodes
- NodeRED : (very low ressources) data viz editor (runs on RaspberryPi)

#### Data visualization tools 

- Programming libs code + tools (covers on the next vid) 
- Hue : creates visualization from SQL queries
- Kibana : data viz for web application based on Elasticsearch
- superset : data exploration & viz for web application

#### Model Deployment tools 

After a model is built => `API consummable` to another apps

- PredictionIO : Apache Spark ML models
- SELDON : supports lots of frameworks (TF, SparkML, R, sci-kitlearn) runs on top of Kubernetes, RedHat OpenShift
- mleap : creates SparkML
- TensorFlow (TF Lite : embedded systems (RPi, Smartphone); TF.JS : Web)

#### Model Monitoring and Assessment Tools 

- ModelDB : a system to manage ML models (storage, quering ), supports : SparkML pipelines and scikit-learn
- Prometheus : multi-purpose tool
/!\ Model perfomance is not EXLUSIVELLY measured through Accuracy & Model bias 
- IBM AI fairness 360 OS Toolkit : detects bias and ml models 
- IBM Adversarial robustness 360 ToolBox : protect model against hacking and covers vulnabilities
- IBM Explanability 360 Toolkit : model description

#### Code Asset Management tools (version mgt)  

- git 
- GitHub
- GitLab
- BitBucket

#### Data Asset Mgt tools

- Apache Atlas : 
- ODPA EGERIA : exchange data
- Kylo

### Open Source Tools for Data Science - Part 2  

#### Development Environment 

- Jupyter : notebook for python programming (data science ide : code, interpreter, results viz)
- JupyterLab : next generation of jupyter nb
- Apache Zeppilin : uses integrated lib for ploting opposite to jupyter nb
- R-Studio  : IDE for data viz for R language, dev in python also possible
- SPYDER : like E-Studio but for Python world


#### Execution Environments (Data Clustering) : to execute a large amount of data

- Apache Spark :  scalable execution performance (mostly used)
- Apache Flink : batch processing engine for RT data streaming
- RAY (riselab) : deeplearning training on a large scale

#### Fully INTEGRATED Visual Tools 

![](./docs/data-science-workflow-tools.PNG)

- No programming skills necessary
=> Data Int - Data Viz - Model Building

- KNIME (open innovation) : created by University of Konstanz in 2004 
- ORANGE

### Commercial Tools For Data Science 
- Data Mgt : 
  - Oracle DB
  - MS SQL Server
  - IBM DB2

- Data int & Transformation (ETL)
  - Informatica 
  - IBM InfoSphere DataStage
  - Talend
  - IBM Watson Studio Desktop

- Data visualization
  - Tableau
  - MS Power BI 
  - IBM Cognos Analyst

- Model Building 
  - SPSS 
  - SAS 

- Data Deployment 
  - IBM SPSS (IBM Watson Studio Desktop)
  - PMLM ? 

- Model Monitoring & Code Asset Mgt : USE OPENSOURCES TOOLS
  - Git & GitHub

- Dev Env : 
  - IBM Watson Studio (Desktop

- Fully Integrated Visual Tools : 
- IBM Watson Studio (Desktop : Jupyter + graphic tools )
- IBM Watson Open Scale
  - Can be deployed on the top of Kubernetes and RedHat OpenShift

=> H2O Driveless AI : covers the completed data science cycle

### CLOUD BASED TOOLS FOR DATA SCIENCE 

takes operations task away from the user

`SaaS` : Software as a Service - cloud provider operates the tool for the client in the cloud (storage, configuration and updates...)

`PaaS` : Platform as a Service

#### **Data Management**  
  - Amazon DynamoDB : NoSQL db for data storage and retrieval  in key-value/document store format(like JSON) 
  - Cloudant : database as a service offering (based on Apache CouchDB)
  - IBM Db2

#### **Data Integration and Transformation(ETL/ELT)** 
     Informatica
     IBM Data Refinery (Part of IBM Watson Studio)

#### **Data visualization** 
- Datameer
- IBM Cognos Analytics
- IBM Data Refinery : data exploration and Viz
	* Word cloud :  document corpus

#### **Model building** 
- IBM Watson Machine Learning
- Google Cloud : AI Platform Training 

#### **Model Deployment** : make
- IBM SPSS Collaboration & Deployement Services : PMML language
- IBM Watson Machine Learning

#### **Model Monitoring and Assessment**
- AWS : Amazon SageMaker Model Monitor
- Watson openScale
- Code Asset Management :
- Development Environment
- Execution Environment :

#### **Fully integrated Visual Tools and Platform** 
- IBM Watson Studio + Open Scale (ML + AI Tasks)
- MS Azure Machine Learning
- H2O AI

### QUIZ : PACKAGES, APIs, DATA SETS and Models

### Libraries for Data Science 
- Libraries(frameworks) : collection of functions and methods that enable to perform a wide variety of actions without writing any code

- Python most useful libs: 
1. Scientific Computing lib
- Pandas : Data stuctures and Tools - cleaning, manipulation, analysis ( data is represeting in row and col model called : DATAFRAME)
- NumPy : Arrays & matrices : (pandas is built on top of NumPy) 
2. Visualization lib 
- Matplotlib : plots & graphs, most popular
- Seaborn :  plots (heatmaps, time series, violin plots ...)
3. high level ML & DL
- Scikit-learn  : ML algorithms like regression,classification, clustering ...)
- Keras : DL and NN 
4. Deep Learning Lib
- TensorFlow : build for production & Deployement
- Pytorch  : build for experimentation 

Apache Spark : general -purpose cluster computing framework to process data using compute-cluster(computing data from multiple computer)
- pandas, Numpy, scikit-learning   
- languages : python, R, scala, SQL 
- Scala libs : Vegas(statical & viz), Big DL (dl)
- R libs(ml & data viz): ggplot2, others (use w/ Keras & TF)   => R is being superseded by python

### Application Programming Interfaces(API)
- API : lets two pieces of software talk to each other (part of library) 
- things to know : input/output 
(program <=> |API| => Other sw components)
- interface between two differents programs/sws (Python => [API] => TF(written in c++))
=>julia/Matlab/Scala/R<=> [API] <=> TF

#### REST(REpresentation State Transport) APIs  : interact w/ web Services
- Applications called through the internet (communication, input/request, Output/Ressources)
- (Client) <=> [API] <=> (Web services)
- Client <=> [API] <=> Ressources(endpoint)
ex : The Watson Speech to Text API / Watson language translator API

### Data Sets- Powering Data Science
- Data Sets : structured collection of data ( text, numbers, or media such as images, audio, or video files)
- 
#### Data Structured :
- Tabular data :  row, col (csv files)
- Hierarchical data, Network data (graph, nodes...)
- Raw files (imgs, audios ..)

#### Data Owenership 
- Private Data
  - Confidential
  - Private or personal information
  - Commercial sensitive
- Open Data : 
  - Scientific institutions 
  - Governments
  - Organizations (UN, OMS)
  - Companies
  - Publicly available

#### Where to find open data
- Open data list : datacalogs.org
- Governmental, intergovernmental & Organizations websites
  - data.un.org
  - data.Governmental
  - europeandataportal.europeandataportal
- Kaggle : kaggle.com/datasets
- Google data set search :  datasetsearch.research.google.com

#### Community Data License Agreement
- cdla.io : Linux Foundation
- CDLA-Sharing
- CDLA-permissive

### Sharing Enterprise Data - Data Asset exchange

- IBM Data Asset eXchange(DAX) : Curated collection of data sets, from IBM Research and 3rd party
  - Diverse data types (imgs, text files) & high level of curation for data set quality
  - Friendly license & use terms
  - notebooks examples (charts, timeseries, ml model)
  - DAX projects on ibm.develpper websites

#### Get Stated w/ DAX
- download dataset or explore w/ notebook on Watson Studio 

### Machine Learning Models - Learning from data to make predictions

- Data can contain a wealth of information
- traditional programming approach reaches its computer limit due to the alarge amount size of data
=> Solution : Machine Learning Model : algorithms which learn from experiences by identifying patterns and predicts the result  
- A model must be trained on data before it can be used to make predicts 
  
Types of ml learning classes : 
- **Supervised** (most commun) : Data is labeled and model trained to make correct predictions (input data(X), corrected output (y))
	examples : 
	- regression : predict real numerical values (home sales prices, stock market)
	- classifications : classify things into Categories (email spams filters, fraud detection, image classification)
- **unsupervised** : data is not labeled. Model tries to identify patterns w/out external help
	- clustering and anomaly detection
- **reinforcement** : Conceptually similar to human learning processes
	- a robot learning to walk, Go, Chess and other strategy games

* List of ML Algorithms :https://www.newtechdojo.com/list-machine-learning-algorithms/


### Deep Learning Models 
- Tries to loosely emulate how human brain works
Applications : 
  - Natural Language Processing
  - Image, Audio, and video analysis
  - Time Series forecasting
  - Much more
  - LARGE datasets of labeled data and compute intensive

- Build from scratch / download from public model repos
- frameworks : TensoFlow, PyTorch, Keras
- populars model repos : 
	- model zoo
	- ONNX model zoo 

**ML/DL Pipeline-process**

```mermaid
graph LR;
  A[Prepare Data]-->B[Build Model];
  B-->C[Training model]	
  C-->D[Deploy model]
```
=> Prepare Data  : (collect data , clean, label)  
=> Build Model : from scratch/ zoo model that fits to the problem
=> Training Model(iterative process ) : requires (more) data, time, expertise, ressources
=> Deploy model : make available for the application

### The Model Asset Exchange (MAX)

- Model build 'Time to value' is a long process and need to be optimized  : MAX does that!!!
- free open source dl microservices
  - pre-trained / custom trainable state of art models 
  - fully tested, deploy 
  - approved personal/commercial use
- Vailable for variety of domains
  - object detection
  - image, audio, text classification(what is in this...)
  - named entity recognition
  - img to Text (generate image caption)
  - human pose detection

### MAX model serving microservices

```
-----------------------------------------------------------------------------------------
Data 				   + 	Model  +	Compute ressources 		+ 		Domain expertise
(Input/output and			(	Pretrained	 model	) 					(REST API)	
model processing code)
-----------------------------------------------------------------------------------------
  model serving container (Docker images ) => deploy in production w/ Kubernetes
----------------------------------------------------------------------------------------
HW : Local Machine/Private cloud / Hybrid cloud / Public cloud
-----------------------------------------------------------------------------------------

```

### Model serving microservices API 
- REST API 
- developer.ibm.com/exchange/models


### Reading: Explore Data Sets and Models 

Model Asset Exchange and the Data Asset Exchange

- Model Asset Exchange (MAX) and the Data Asset Exchange (DAX) 
- open source Data Science resources on IBM Developer.

#### Exercise 1: ex ref : https://colab.research.google.com/drive/1ijvJh9FRhmfQzHgtPexnQMPNlowfluwz?authuser=1#scrollTo=Mo60qFIu7pHl
Where to find open data sets on IBM Developer
How to explore those data sets


#### Exercise 2: http://ml-exchange.org/models
Find ready-to-use deep learning models on the Model Asset Exchange
Explore the deep learning model-serving detecting the image 

## **W2 : Open Source Tools**

### Introduction to Jupyter Notebook  : 

- Jupyter : Julia-Python-R
- jupyter created from ipython 
- A tool for recording Data Science experimentation
- It allows Data Scientist to combine text and code block in a signle file
- It generates plots and tables within the file
- Notebooks can be exported as pdf and html files

JupyterLab : open multiple jupyter files

Jupyte Environment
- Google Collab
- IBM 

#### Installation : 

```pip install JupyterLab ``` or from ANACONDA 

SKILLS NETWORK : virtual jupyter notebook nothing to be installed

virtual jupyter notebook : https://labs.cognitiveclass.ai/tools/jupyterlite/lab/tree/labs/DS0105EN/Jupyter_Notebook.ipynb?lti=true

### Getting Started w/ Jupyter
- open on SKILLS NETWORK
- slide mode to deliver the work
- terminate the session (before leaving the jupyterlab ? ) 

### Jupyter Kernels
- A notebook kernel : computational engine that excutes the code contained in a Notebook file
- Jupyter Kernels for many other languages exist
- the kernel perfoms the computational and produces results, when the notebook is executed
- Other notebook languages can be need based on the Environment

### Jupyter Architecture
- implements a 2 process model : kernel - client
- client : interface enabling the user to send code to the kernel
- Kernel : execute the code and returns to the client for display
- the client is the browser when using jupyter notebook

(user) -> (browser) -> (Notebook server) -> (Kernel) 
						|
					(Notebook file)

NB Converts : Converts notebook to the new file format (pdf, html)	

### Lab
- Hands-on Lab: Jupyter Notebook - The Basics
- Ungraded External Tool: Ungraded External ToolLab - Jupyter Notebook - The Basics
- Lab - Jupyter Notebook - More Features
- Ungraded External Tool: Ungraded External ToolLab - Jupyter Notebook - More Features
- Hands-on Lab: Jupyter Notebook - Advanced Features
- Ungraded External Tool: Ungraded External ToolLab - Jupyter Notebook - Advanced Features

### Jupyter Notebooks on the Internet
- First you start with exploratory data analysis, so this notebook is highly recommended to have a look at:
 https://nbviewer.jupyter.org/github/Tanu-N-Prabhu/Python/blob/master/Exploratory_data_Analysis.ipynb

- For data integration / cleansing at a smaller scale, the python library pandas is often used. Please have a look at this notebook: 
https://towardsdatascience.com/data-cleaning-with-python-using-pandas-library-c6f4a68ea8eb

- If you want to already experience what clustering is, have a look at this: 
https://nbviewer.jupyter.org/github/temporaer/tutorial_ml_gkbionics/blob/master/2%20-%20KMeans.ipynb

- And finally, if you want to go for a more in-depth notebook on the iris dataset have a look here:
 https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis

### Quiz - Jupyter Notebook

### RStudio IDE

### Introduction to R and RStudio
- Statistical programming Language
- Used for data processing and manipulation
- Statistical, Data Analysis and Machine Learning
- R is used ùost academics, healthcare and the gouvernment
- R supports importing data from different sources : 
  - Flat files
  - Databases 
  - Web
  - statistical sw : SPSS, STATA ...

#### R capabilities : 
- ease to use
- great for data viz
- Basics analysis doesnt need any packages tobe installed

#### RStudio IDE 

- Code editor
- Console
- Workspace/History tabs 
- Plots, files , helps, packages ....

#### R libs : 

- dplyr : Data Manipulation
- stringr : String Manipulation 
- ggplot : Data Visualization 
- caret : Machine Learning 

#### RStudio Virtual Environment  : 
- [RStudio Virtual Environment ](https://labs.cognitiveclass.ai/tools/rstudio-ide)
  
        ### Plotting within RStudio
        - ggplot : histograms, charts, scatterplots
        - plotly : web
        - Lattice : complex, multivariables datasets  
        - Leaflet : interatives plots

        #### installing lib  
        install.packages("package name")

        #### using a lib 
        ex : library(ggplot)

### Ungraded External Tool: Getting started with RStudio and Installing packages
- [Getting started with RStudio and Installing package](https://labs.cognitiveclass.ai/tools/rstudio-ide)
- 

### Getting started with RStudio and Installing packages

    - install.packages("package name")
    - # install.packages("GGally", repos = "https://cran.r-project.org", type= "source") 
    - the packages are located at  : '/usr/local/lib/R/site-library'

### Ungraded External Tool: Ungraded External ToolPlotting within RStudio
- Practice

### Plotting within RStudio

### get details of dataset : '?mtcars'

### Ungraded External Tool: Ungraded External ToolPlotting within RStudio (Advanced)

# Quiz - RStudio IDE	


### GitHub

### Overview of Git/GitHub
- Git is a Version Control System tool
- Version Control System  : keep track of changes of a document and make collaboration much easier
- Git : Free and open source sw (GPL)
- Distributed version control system
- Accessible anywhere in the world 
- Most popular vcs outthere

#### Git (client) + GitHub (server : web-hosted) : 
- GitLab
- GitBucket 
- Beanstalk 

#### SHORT Glossay : 
- SSH protocol : A method for secure remote login from one computer to another
- repository : folders of a project that are set up for version control
- Fork : a copy of a repo
- Pull request : The process to request someone reviews and approves your changes before they become final
- Working directory : A directory on your file system, including its files and subdir, that is associated with a git repo
- Git commands : https://git-scm.com/docs
- GitHub : https://try.github.io

### Lab 1: GitHub Lab - Getting Started
- done
### GitHub - Working with Branches
- snapshot of a repo
- main branch(revied and tested code) vs child (experiment, new features) branch
### Lab: Branching, Merging and Pull Requests on GitHub (Optional)
- done 
### Git and GitHub via command line (Optional)
- git init, git status, git add, git commit ...
- Pre-requisites for command line interface (Optional)
-  Configuring SSH access to repository (Optional)
-  Git and GitHub via command line instructions (Optional)
-  Branching and merging via command line (Optional)
-  Lab 2: Branching and merging via command line (Optional)
-  Contributing to repositories via pull request (Optional)
-  Lab 3: Contributing to repositories via pull request (Optional)
-  Practice Quiz: Practice Quiz - GitHub
-  3 questions


## **W3 : Watson Studio**

### What is IBM Watson Studio?
- **DATA : greater ressources of every smart Organization**
- IBM IDE for DATASCIENCE and DATA PROJECTS
- data analysis, model building, data viz, open sources and collaborative  framework 

- Applications : 
  - innovation
  - Data Refinery
  - Dev and traniing of ML/DL models 

### Watson Studio Introduction

 - A effecient Data driven Team : 
   - Data Engineer
   - Data Scientist
   - Data Steward
   - Business Analyst
   - Developer

 - collaborative and easy to create a project (feature) : Overview, Asset, Env, Jobs ...


### Creating an Account on IBM Watson Studio
- need to use the bank account to finish the registration - it sucks ! 

### Jupyter Notebook in Watson Studio - Part 1
ASSET
- Add a notebook to the project (*.ipyn file)
- add a dataset to work with and open the data as pandas dataframe
- Add a description : add cells etc ... & run the code 
- Create a Job so that the nbs run in different time

### Jupyter Notebook in Watson Studio - Part 2
* ENVIRONMENT SET UP * 
- sw & hw configuration 
- Default or Spark (for additional infos)
- Choose the sw version (Python, R, Scala ..) 
- Stop the Kernel to start the new env, associate w/ nb
- Publish the nb to the GitHub

### Ungraded External Tool: Ungraded External ToolObtain an IBM Cloud Feature Code
- created new account and added a new code

### Hands-on Lab: Creating a Watson Studio Project with Jupyter Notebook
- Task 1: Create Watson Studio Service:
- Task 2: Open Watson Studio
- Task 3: Create a Project
- Task 4: Adding a Notebook to the Project
 => Add assets => JNB editor

Linking GitHub to Watson Studio
- create a Github token 
- generate a token 
- settings : Integration => Github Working Repo

### Practice Quiz: Practice Quiz - Watson Studio

## Other IBM Tools

## Other IBM Tools for Data Science : intro

### IBM Watson Knowledge Catalog 
=> Catalog and manage all data ressources <= 
- unites all Catalogs in a single metadata-rich Catalog
- corresponds  : Data Asset Mgt, Code Asset Mgt, Data Mgt, Data Integration & Transformation
- Main features : Find data, Catalog data, Govern, Understand, Power data science, Prepare and Connect, Deploy Data 
- build on the top of RedHat OpenShift
- DC : protects data from misuse and enables sharing of assets automatically and dynamically masking sensitive data
- runs on the private or public cloud
- Catalog  : metadata about contents of assets and their access
- A metadata is encrypted and store in the secure cloud location 
- Storage places : IBM Cloudant, Db2, Db2 Warehouse, AWS, Azure, twitter, pdf ...
- Need a permission to access a specific/sensitive catalog 
- IBM Watson Studio  : There are some public Catalog 


### Data Refinery
=> graphical tools for analyzing and preparing data <=
- Power tool for data preparation and analysis
- Simplifying Data Preparation 
- Interative visual interface that enables self-service preparation
- available in Watson Studio
- analyse the data and run at the end in th real dataset
- relational operators  : left join, inner join etc
- automation (from database ) => (Data refinerey) => Target table


### SPSS Modeler Flows in Watson Studio
=> easy to use graphical interfaces for statistics, ML and ETL <=
- Data Mgt, Data Int & Trans, Data Viz and Model Building
- build ML and pipeliness (SPSS Modeler)
- Model objects : data sources(inputs), type, agregate, filter, merge 

#### Types of nodes :
- Palettes(menu on the left) and canavas(main part o fthe screen)
- Data sources are located in the import
- After the model setup and configuration => RUN => Generate Model (in Nuggets)

### SPSS Modeler Flows : 
- Each flow starts w/ data sources located in the import group
=> Ex: AI dataset (drugs from Watson Studio env)
- Target(label) => Categorical field
- Predictive variables 

#### Examing models
- Predict values (Categories)
- Model accuracy
- Confusion matrix
#### Network Diagram 
- NN represent of the built model ( input layers, hidden layers and output layers)
- weights and theirs values, and nn connections  ...

#### Auto-Classifer and Auto-NUmeric Nodes: 
- Categories and continuos target perspective
- build severals models and pick the best one based on a certain criterion 

# Lab: Modeler Flows in Watson Studio : assumed as done due to the access problem

# IBM SPSS Modeler 
- create by Clementine 1994
- acquired by SPSS in 1998 
- SPSS acquired by IBM 2009
- Building model w/ coding (Boxes and objects, graphics)
- Data Mgt, Data Int & Trans, Data Viz and Model Building + Model Deployement

## Prediction a CHURN (from telecommunication database)
- round node
- hexagone node (data sources)
- feature selection node => build to generate a yellow nugget 
- Super Node (special node created by the user) 
## Build and Examine
- run the flow
##  Examine output (on trained data)
- input and target
- classification table which resumes the model features
## Training and test
- Partition node : records for testing and validation 
- partitioned data ( to avoid Overfitting - high accuracy)
## Model prediction(scoring) & evaluation (confusion matrix and Accuracy) 
- introduction to IBM SPSS Modeler:  https://www.youtube.com/watch?v=bJfe9C9_hjY

# SPSS Statistics (1968 )
- IBM SPSS Statistics & ML application
- build predictive model, statistical analysis of data
- build statistical & Data mining algos w/out coding 
- file format (*.sav)
- Load dataset are presented as structured data table (excel )
## useful MENUS : 
- DATA : 
- TRANSFORMATION : 
- ANALYZE : Chose the type of model to be applied : ex: Tree Model
- show model detailed after building (classification table)
- GRAPHS  : data viz : chart builder
- SPSS Synthax : special programing language used by others tools

Learn SPSS : https://www.youtube.com/watch?v=TZPyOJ8tFcI

### Model Deployment with Watson Machine Learning
- The return of investment is obtained when the model/pipeline is put into production
- To make prediction, scores and new cases ..
#### standard for model Deployement : 
- Sage Maker (Amazon)
- MLFlow (Databricks)
- AirFlow (Airbnb)
- Kubeflow (Google)
#### PMML : Predictive Markup Model Language
=> created by Data Mining Group(DMG) in 1990s
- Generated from : Watson Studio & SPSS
-> PFA Portable Format for Analytics from DMG
-> JSON => the revolution !!! (2013)
#### ONNX : Open Source DL project from MS & FB
- models are built for very specific Environment (embedded device : GPU, CPU, FPGA ..)
#### IBM  Watson ML
- support  : SPSS Modeler, PMML & ONNNx

### Auto AI in Watson Studio
- create ML Pipelines !!!
- Help to simplify an AI lifecycle management, AutoAI automates : 
- AutoAI Process/pipeline : 
(Raw labeled Dataset)=>(Preparation)=>(Model selection)=>(HPO)=> (Feature engineering)=>(HPO)
- Build pipelines, comparing them according to the adverses experiments...
- Classification & regression ONLY 
- Pipelines are saved as ML assets in Watson Studio

### IBM Watson OpenScale (create ML Pipelines)
- Fairness 
- Explanability : Enhances compliance w/ regulation  
	=> Fair Credit Reporting Act 
	=> GDPR
- Model Monitoring (bais in the prediction (%) ...)
	=> control and increase Accuracy and fix inconsitency
- Business Impact   

### Practice Quiz: Practice Quiz - Other IBM Tools : 12 questions

## **W4 : Create and Share Your Jupyter Notebook**

Instructions: Create and Share Your Jupyter Notebook
- [My Project link](https://eu-de.dataplatform.cloud.ibm.com/analytics/notebooks/v2/7ca3c478-e3f7-4bbe-9ddb-4da5bdb3dfb2/view?access_token=bca88978829370aace11414003eca738683d4e720317bde195d1752d867860e8)

### Grade : 
- Peer-graded Assignment: Submit Your Work and Grade Your Peers
- Review Your Peers: Submit Your Work and Grade Your Peers
- Reading: Reading IBM Digital Badge


## References 
- [Course-2 : Tools for Data Science](https://www.coursera.org/learn/open-source-tools-for-data-science?specialization=ibm-data-science)
- [Mermaid doc](https://unpkg.com/mermaid@7.0.3/dist/www/flowchart.html#nodes-shapes)


