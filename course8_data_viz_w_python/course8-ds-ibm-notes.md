# Course 8: Data Visualization with Python 

**What you'll learn**

- Implement data visualization techniques and plots using Python libraries, such as Matplotlib, Seaborn, and Folium to tell a stimulating story
- Create different types of charts and plots such as line, area, histograms, bar, pie, box, scatter, and bubble
- Create advanced visualizations such as waffle charts, word clouds, regression plots, maps with markers, & choropleth maps
- Generate interactive dashboards containing scatter, line, bar, bubble, pie, and sunburst charts using the Dash framework and Plotly library

**Skills you'll gain:** Data Science, Data Analysis, Python, Programming, Pandas, Jupyter notebooks


## Table of Contents

- [W1: Introduction to Data Visualization](#w1-introduction-to-data-visualization)
- [W2: Basic \& Specialized Visualization Tools](#w2-basic--specialized-visualization-tools)
- [W3: Advanced Visualization \& Geospatial data](#w3-advanced-visualization--geospatial-data)
- [W4: Creating a Dashborad with Plotly and Dash](#w4-creating-a-dashborad-with-plotly-and-dash)
- [W5: Practice Assignment](#w5-practice-assignment)
- [References](#references)


## W1: Introduction to Data Visualization

# Syllabus
# Welcome
# Introduction to Data Visualization (Module 1)
## Main reason why data viz is important :  
	=> For exploratory data analysis
	=> Communicate data clearly
	=> Share unbiased representation of data
	=> Use them to support recommendation to different stakeholders
## Best practices : 
- Darkhorse Analytics : created in 2008 in (university of alberta)
- quantitative data consulting
- data viz & geo spatial analysis 
- data viz approach : a plot is meant to get accross NOT distractive
	1. Less is more effective
	2. Less is more attractive
	3. Less is more impactive 
- Goal : have a graphic easier to read, cleaner, less distractive, better to understand ..  
## References : https://www.darkhorseanalytics.com/

# Introduction to Matplotlib
- Most popular data viz library 
- Created by John Hunter(Neurobiologist)
- Initially created for EEG/ECoG Visualization Tool 
- Inspired from MATLAB 

## Matplotlib Architecture 

```
+--------------------------+
| Scripting Layer(pyplot)  |
+--------------------------+
+--------------------------+
| Artist Layer(artist)     |
+--------------------------+
+--------------------------+
| Backend Layer(FigCanvas..|
+--------------------------+
```

### Backend Layer(FigCanvas, renderer, events ...)
1. FigureCanvas : matplotlib.backend_bases.FigureCanvas 
	=> Encompasses the area onto which the figure is drawn
2. Renderer :  matplotlib.backend_bases.Renderer
	=> Knows how to draw on the FigureCanvas
3. Event : matplotlib.backend_bases.Event 
	=> Handles user inputs such as keyboards strokes and mouse clicks

### Artist Layer(artist)
- Comprised of one main object - *Artist*
	=> Know how to use the Renderer to draw on the Canvas
- Responsible for : Title, lines, tock labals, and images, all correspond to individuals Artist instances 
- 2 types of artist objects : 
	=> 1. Primitive : Line2D, Rectangle, Circle, and Text 
	=> 2. Composite : Axis, Tick, Axes, and Figure
- Each composite artist may contain other composite artists as well as primitive artists

### Scripting Layer
- Comprised mainly of pyplot, a scripting interface that is lighter that the Artist layer 
- Let's see how we can generate the same histogram of 10000 random values using the pyplot interface 

### References : aosabook.org/en/matplotlib.html 

# Basic Plotting with Matplotlib
- Support by differents env : Python scripts, iPython shell, web app & servers ... jupyter nb
- "Dynamic" ploting using BACKENDS : modify plot, costomize ...
- use magic functions(%) to execute backend objects 
	=> magic function starts w/ % sign
- Some backends : 
	=>%matplotlib inline : plot window within the browser and not in separeted window
	=>%matplotlib notebook : allows to modify figure once is RENDERED !
- Matplotlib - PANDAS 
	=> df.plot(kind="line")
	=> df["x"].plot(kind="hist")

# Dataset on Immigration to Canada
- Dataset : 
	=> src : United Nations (45 countries)
	=> annual data on the flows of international migrants
	=> migrants to canada dataset 

- Import the dataset with pandas
- to check the imported data : df.head, df.describe ...

# Line Plots
- continuos dataset  
- plot which displys information as series of data points called "markers" connected by the straight line segments
- using pandas dataframe, map function to create an iterative point per axis value 

# Lab
# Quiz

## W2: Basic & Specialized Visualization Tools

# Area Plots
- Also known as 'area chart / graph'
- commonly used to represent cumulated totals using numbers or percentages over time
- based on the 'line plot'
- Dataset-processing steps : 
=> index : country (since we want to know/focus on canada)
	=> now each represents a country
=> add a 'total() column (number of total(sum of population/immigrants) per country)
	=> sorting by descending or order allows to get highest population per country
=> df_canada (Dataframes Canada specified )
=> Fix index problems : transposition axis, sort_values ...
- generate "area plot"  : plot(kind='area')

# Histograms
- way of representing the frequency distribution of a variable 
- To generate "histogram"  : plot(kind='hist')
- To correct horizontal axis problem : split the bins(ticks) with numpy (np.histogram())
	=> then use theses bins(values) to generate the histogram w/ matplotlib 

# Bar Charts
- Unlike a histogram, a bar chart is commonly used to compare the values of a variable at given point in time.
- generate "bar plot"  : plot(kind='bar')

# Lab
# Quiz

# Pie Charts
- graphs divided into slices to illustrated numerical proportion
- generate "pie plot"  : df.plot(kind='pie')
- Cons : fail to represent data in a consistent way and getting the data accros(presentable, understandble, clear, visible ...)  

# Box Plots
- Way of statistically representing the distribution of given data 
through 5 main dimensions : 
	1. min == (IQR - IQR*1.5) (Inter Quartile Range)
	2. 25% (1st IQR)
	3 50% = median
	4. 75% (3/4 of sorted data)
	5. max (75% + IQR*1.5)
- display outliers as individuals dots that occur outside the upper and lower extremes
- generate "pie plot"  : df.plot(kind='box')

# Scatter Plots (nuage des points)
- displays relationship datas btw 2 variables (x, y) to determine if any correlation exists
- generate "scatter plot"  : df.plot(kind='scatter', x='df_x', y='df_y')  

# Lab
# Quiz

## W3: Advanced Visualization & Geospatial data

# Waffle Charts
- interesting visualization that is normally created to display progress toward goals
- the more the contribution the more tiles(visibility) 
- matplotlib doesn't have a built-in function to generate waffle charts 

# Word Clouds
- A depiction of the frequency of different words in some textual data 
- the more specific the word appears in the source of textual data the bigger/bolder in word cloud 
- Matplotlib doesn't have a function/module to generate word clouds but Python does! 
- Andreas Mueller (created a python lib for cloud word generation)
- usage => Lab session

# Seaborn and Regression Plots
- Python visualization library based on matplotlib
- provides high level interface for drawing attractive statisticals graphics : regression plots, box plots ...
- less lines of codes to plot than matplotlib

# Lab
# Quiz

# Introduction to Folium
- Powerful Python library that helps you create several types of Leaflet maps 
- First created to visualize Geospatial data 
- locate any word based on : [latitude, longitude] values 
- It enables both the binding of data to map for choropleth visualizations as well as passing visualizations as markers on the map
- The library has a number of built-in tilesets from OpenStreetMap, Mapbox, and Stamen, and supports custom tilesets with Mapbox API keys. 
## Creating a World Map
- word_map = folium.Map()
- It's possible to get a specific map place/location(latitude/longitude)
- Also possible to add different styles of maps ...

# Maps with Markers
- create a feature group
	=> add children (w/ the location of the place you want to show)
	=> add markers 
	=> markers can be labeled 

# Choropleth Maps
- Thematic map in which areas are shaded or patterned i proportion to the measurement of the statistical variable being displayed on the map : 
	=>  poplulation density or per capita income ...
	=> The higher the *measurement* the darker the color 
- this graph required : Geojson to be setup (json file with some metadatas)
## Creating the Map : 
- world_map.choropleth() + world_geo_json_path

# Lab
# Quiz

## W4: Creating a Dashborad with Plotly and Dash

# Module Overview and Learning Objectives
# Dashboarding Overview
- Interactive Data Application (IDA) can help improve business performance
## Dashborad : 
=> Realtime visuals
=> Understand business moving parts
=> Visual track, analyze, and display key performance indicators(KPI)
=> Take informed decisions and improve performance 
=> Reduced hours of analyzing
- Best dashboards answer important business questions
## (Practical) Scenario
- Monitor and report US airline performance 
=> Requested report items 
1. Top 10 airline carrier in year 2019 in terms of number of flights 
2. Number of flights in 2019 split by month 
3. Nb of travelers from California (CA) to other states split by distance group
- Presenting results in Table and Documents is time consumming, less visual, and difficult to understand ...
- A datascientist should be able to create and deliver a story around the *finding* in a way a stakeholders can easily understand 
	=> Datashboards are the way to go !!! 
## Tools : Web-based Dashboarding
- Dash from Plotly 
	=> Python framework for web analytic applications
	=> written in the top of Flask, Plotly.js and React.js 
	=> data viz apps with highly custom UI 
- Panel 
	=> works with visualizations from Brokeh, Matplotlib, HoloViews ... 
- Voilà : turns jupyter nb into standalone web app 
	=> jupyter-flex 
	=> template : voila-vuetify
- Streamlit : datascripts into shareable web apps w/ 3 rules 
	=> embrace python scripting 
	=> treat widgets variables
	=> reuse data and computation

## Tools : Dashboarding
- Bokeh
- ipywidgets 
- matplotlib
- Bowtie
- Flask (dashboard)

# Additional Resources for Dashboards
https://pyviz.org/dashboarding/index.html
https://www.theguardian.com/news/datablog/2013/mar/15/john-snow-cholera-map

# Introduction to Plotly
- Interactive, open-source plotting library
- Supports over 40 unique chart types 
- Includes chart types like statistical, financial, maps, scientific, and 3-dimensional 
- Visualizations can be displayed in Jupyter notebook, saved to HTML files, 
or can be used in developing Python-built web applications

## Plotly Sub-modules
- Plotly Graph Objects : Low-level interface to figures, traces, and layout
	=>plotly.graph_objects.Figure (high level)

- Plotly Express : High-level wrapper
	=> More simple syntax
	=> uses graph objects internally
## usage : 
- fig = go.Figure(data=go.Scatter(x=x, y=y)) // go :graph_objects
- fig = px.line(x=x, y=y, title='Simple Line Plot', labels=dict(x='Month', y'Sales')) // px : plotly.express

# Additional Resources for Plotly
https://plotly.com/python/getting-started/
https://plotly.com/python/graph-objects/
https://plotly.com/python/plotly-express/
https://plotly.com/python-api-reference/
https://images.plot.ly/plotly-documentation/images/plotly_js_cheat_sheet.pdf
https://community.plotly.com/c/plotly-python/5
https://plotlygraphs.medium.com/
https://developer.ibm.com/exchanges/data/

# Lab : Plotly basics: scatter, line, bar, bubble, histogram, pie, sunburst

# Introduction to Dash
- Open source user interface Python libray from Plotly 
- interactive web based apps 
- web server running Flask and communicating JSON packets over HTTP requests 
- Dash frontend renders uses React.js 
- Easy to build GUI 
- Declarative and Reactive 
- Rendered in web browser and can be deployed to servers 
- Inherently cross-platform and mobile ready 
## Dash components
- Core components
	=> import dash_core_components as dcc
- HTML Components 
	=> import dash_html_components as html

# Theia Labs Overview
# Dash basics: HTML and core components
# Additional Resources for Dash
https://dash.plotly.com/
https://dash.plotly.com/dash-core-components
https://dash.plotly.com/dash-html-components
https://community.plotly.com/c/dash/16
https://medium.com/plotly/tagged/dash

# Make dashboards interactive
- connect core and html components 
	=> Dash - Callbacks : python function that are automatically called by Dash whenever an input component's property changes 
	=> Callback function: is decorated @app.callback decorator 
	=> when the input component value changes the callback function wrapped by the decorator update the output 
	=> Callback w/ 1,2 ... inputs 
# Additional Resources for Interactive Dashboards
https://realpython.com/primer-on-python-decorators/
https://peps.python.org/pep-0318/#current-syntax
https://dash.plotly.com/basic-callbacks
https://dash.gallery/Portal/
https://plotly.com/dash-community-components/


# Lab
## Add interactivity: user input and callbacks
## Flight Delay Time Statistics Dashboard
# Quiz 

**Lesson Summary**

- Best dashboards answer critical business questions. It will help business make informed decisions, thereby improving performance. 
- Dashboards can produce real-time visuals. 
- Plotly is an interactive, open-source plotting library that supports over 40 chart types. 
- The web based visualizations created using Plotly python can be displayed in Jupyter notebook, saved to standalone HTML files, or served as part of pure Python-built web applications using Dash. 
- Plotly Graph Objects is the low-level interface to figures, traces, and layout whereas plotly express is a high-level wrapper for Plotly. 
- Dash is an Open-Source User Interface Python library for creating reactive, web-based applications. It is both enterprise-ready and a first-class member of Plotly’s open-source tools. 
- Core and HTML are the two components of dash. 
- The dash_html_components library has a component for every HTML tag. 
- The dash_core_components describe higher-level components that are interactive and are generated with JavaScript, HTML, and CSS through the React.js library. 
- A callback function is a python function that is automatically called by Dash whenever an input component's property changes. Callback function is decorated with `@app.callback` decorator. 
- Callback decorator function takes two parameters: Input and Output. Input and Output to the callback function will have component id and component property. Multiple inputs or outputs should be enclosed inside either a list or tuple. 

## W5: Practice Assignment


## References

Additional Resources for Dashboards:

- https://pyviz.org/dashboarding/index.html
- https://www.theguardian.com/news/datablog/2013/mar/15/john-snow-cholera-map

