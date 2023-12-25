# Course 6: Databases and SQL for Data Science with Python 

**What you'll learn**

- Analyze data within a database using SQL and Python.
- Create a relational database and work with multiple tables using DDL commands. 
- Construct basic to intermediate level SQL queries using DML commands. 
- Compose more powerful queries with advanced SQL techniques like views, transactions, stored procedures, and joins. 

**Skills you'll gain:** Python Programming, Cloud Databases, Relational Database Management System (RDBMS), SQL, Jupyter notebooks


## Table of Contents

- [Module 1: Getting Started with SQL](#module-1-getting-started-with-sql)
- [Module 2: Introduction to Relational Databases and Tables](#module-2-introduction-to-relational-databases-and-tables)
- [Module 3: Intermediate SQL](#module-3-intermediate-sql)
- [Module 4: Accessing Databases using Python](#module-4-accessing-databases-using-python)
- [Module 5: Course Assignment](#module-5-course-assignment)
- [Module 6: Bonus Module: Advanced SQL for Data Engineers (Honors)](#module-6-bonus-module-advanced-sql-for-data-engineers-honors)
- [Lab](#lab-3)
- [References](#references)


## Module 1: Getting Started with SQL

### Basic SQL: Welcome to SQL for Data Science

## Why SQL for Data Science
- Median based salary(wage) : 110k
- job satisfaction score : 4.4/5
- Top spot on Glassoor's best jobs in America 
- enables to communicate with databases
- every app needs to store data sommewhere : 
=> big data
=> Table with a few rows
=> Small start up
=> Big Database

* Advantages : 
** Boost your professional profile
** Good understanding of relation databases
** SQL stataments are FAST

# About Hands-on Labs in this course : OK

# Introduction to Databases

=> ref : Course 2 : TOOLS FOR DATA SCIENCE => SQL
## data ? : 
=> Facts (words, numbers)
=> Pictures
=> One of the most critical assets of any business
=> /!\ need to be secure & STORE => DATABASE

## Database(db) ? 
- A repository of data
- prodives the functionality for adding, modifying and querying the* data
- differents kinds of db, sotres data in different forms

## Relational database
- Relational DB (collection of two dimmensional tables like row, column) : 
=> relationship between table 

## DBMS (Data Base Management Systems):
- sw to manage dbs

## RDBMS (Relational Data Base Management Systems):
- controls : access, organization, storage ...
- uses in many industries  :  bank, tansportation, health ...
- tools : MySQL, Oracle DB, IBM Db2 ..

## Basic SQL Commands (CRUD)
- Create a table 
- Insert
- Select(read/retrieve)
- Update 
- Delete 

# SELECT Statement (query)
- See/Retrive row from a table
- select statement : query
- result from the query : Result set/table
- cmd : 
=> select* from <tablename>

- retrieve subset or specific column
=> SELECT <column 1>, <column 2> from <tablename> 

## Restricting the result Set : WHERE Clause 
- Restricts the result set
- Always requires a Predicate : 
	=> Evaluate to: True, False or unknown
	=> Used in the search condition of the Where clause
Ex: select <table_id> , title from <tablename>
	WHERE <table_id>='COLOMN_LABLE'

# SELECT statement examples
- IBM Dev Lab notes 

# Hands-on Lab: Simple SELECT Statements
# Ungraded External Tool: ok 
# COUNT, DISTINCT, LIMIT

- COUNT :  retrieves the number of rows matching the query criteria
statement(query)  : select COUNT(*) from tablename
- DISTINCT:  remove duplicate values from a result set
=> retrieve unique values in a columns :
statement(query)  : select DISTINCT columnname from tablename
- LIMIT: retrieves the number of rows retrieved from the database
statement (query) : select * from tablename LIMIT 10 => like head function in pandas

Hands-on Lab: COUNT, DISTINCT, LIMIT
Ungraded External Tool : ok


# INSERT Statement (DML)
- Create the table (CREATE TABLE Statement)
- Populate table w/ data
	=> add new rows to a table (INSERT Statement)
	=> INSERT INTO [tablename]
		<([ColumnName], ...)>
		VALUES([Value], ...)
		...
		([Value], ...) => Multiple rows
--
	INSERT INTO table_name (column1, column2, ... )
	VALUES (value1, value2, ... );
--

# UPDATE and DELETE Statements (DML)
- UPDATE : update a row value
	=> UPDATE [tablename]
		SET[[ColumnName]=[Value]]
		SET([Value], ...)
		<WHERE [Condition]>

--
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
--

- DELETE :  remove a row value 
	=> DELETE FROM [tablename]
		<WHERE [Condition]>

--
DELETE FROM table_name
WHERE condition;
--

Hands-on Lab: INSERT, UPDATE, and DELETE
Ungraded External Tool

Practice Quiz
Practice Quiz
Graded Quiz: Basic SQL
Quiz•3 questions

///W2 : Introduction to Relational Databases and Tables///

# Relational Database Concepts
- Most used data model
- Allows for data independence (logical data, physical data physical storage)
- Data is stored in a tables
- Entity Relationship Diagram (ERD) :
=> Entity : table (object) 
=> Relationship : dependencies

## Entity Relationship Model (ERM) :
- collections of entities
- tool to design relation databases : data type, dependencies... 
- Entity : person, noun, place ...
=> attributes : charateristics of the entity or data elements

## Mapping Entity Diagrams to Tables
- Table : combination of rows and columns
- Entities => tables
- attributes get translated into columns
 => some data values to each of columns the completes the table form

## Primary Keys and Foreign Keys
Primary Keys : unique/id value to prevent duplication of data
Foreign Keys : to create a link w/ others tables/entities


# How to create a Database instance on Cloud
- Cloud databases : database server build / accessed throught a CLoud platform
- ease of use and acess
=> API
=> Web Interface  
=> Cloud ou Remote Applications
- Scalability & Economics
=> Expand/Shrink Storage & Compute resources
=> Pay per use

- Disaster Recovery
=> Cloud Backups

## Cloud databases
- IBM Bd2
- Databases for PostgresSQL
- Oracle Database CLoud Service
- MS Azure SQL Database
- Amazon Relational Database Services (RDS)

Available as : 
=> VMs or Managed Service
=> Single or Mini-tenant

## Database service instances
-  DBaaS provides users access to Database resources in cloud w/out setting up and installing sw
- Database service instance holds data in data objects/tables
- use web interface & API => to get the data once data is loaded

(user API/web interface) ===(query)===> (CLoud DB)
				||	<====== Resultset=====||

## IBM Db2 database on cloud
- On ibm cloud catalog => Db2 services

# Ungraded External Tool : Obtain an IBM Cloud Feature Code
# Hands-on Lab: Create Db2 service instance and Get started with the Db2 console

# Types of SQL statements (DDL vs. DML)
## SQL Commands "groupe" classes?

### DATA DEFINITION LANGUAGE (DDL)
=> Defining Database Objects(define, change, drop data) : CREATE, ALTER, DROP(delete tables), RENAME, TRUNCATE(donot delete the table)

### DATA MANIPULATION LANGUAGE(DML) => CRUD operations
=> Updating database content(read and modify data) : INSERT, UPDATE, DELETE
=> Database searches : SELECT

### DATA CONTROL LANGUAGE(DCL) 
=> Assurance of data integrity and security : GRANT, REVOKE, COMMIT, ROLLBACK...
 

# CREATE TABLE Statement
DDL statement for creating entities(tables) in a relation db
- Syntax : 

CREATE TABLE table_name
(
	column_name_1 datatype optional_parameters,
	column_name_2 datatype ,
	...
	column_name_n datatype ,
);

where : optional_parameters => constraints : Primary key, Foreign key ..

# ALTER, DROP, and Truncate tables

-- ALTER : 
=> add or remove columns
=> modify the datatype of the column
=> add or remove a Keys
=> add or remove a constraints 

- Syntax :
=> add a column
ALTER TABLE <table_name>
	ADD COLUMN <column_name_1> datatype 
	...
	ADD COLUMN <column_name_n> datatype

=> modify a type
ALTER TABLE <table_name>
	ALTER COLUMN <column_name_1> set DATA TYPE 
<new_datatype>

=> delete a column
ALTER TABLE table_name
	DROP COLUMN telephone_number;

=> delete a table
ALTER TABLE table_name; 

=> delete the row values of table without deleting the table 'object'
TRUNCATE TABLE table_name
	IMMEDIATE;

# Examples to CREATE and DROP tables
# Hands-on Lab: CREATE, ALTER, TRUNCATE, DROP
# (Optional) Hands-on Lab : CREATE, ALTER, TRUNCATE, DROP
# Ungraded External Tool•. Duration: 1 hour1h
# Hands-on Lab: Create and Load Tables using SQL Scripts
# (Optional) Hands-on Lab: Create and Load Tables using SQL Scripts
# Ungraded External Tool•. Duration: 1 hour1h
# Summary & Highlights
# Practice Quiz
# Practice Quiz•5 questions
# Graded Quiz: Relational DB Concepts and Tables
# # Quiz•3 questions
# •Grade: --> All good

///W3 : Refining your Results///

# Using String Patterns and Ranges
- Syntax :
SELECT <COLUMN> from TABLE_NAME
WHERE <columnname> LIKE <string pattern>  

<string pattern>   : %(wild card charater) => use before/after the missing letter
between ... and : intervals/range of numbers
in ('values1', 'values2') : where valuesx => are columns

# Sorting Result Sets
## ORDER BY clause(column) => sort value in ascending order
## ORDER BY clause(column) DESC => sort value in descending order
## ORDER BY 2 => sort specific column

# Grouping Result Sets
- GROUP BY : 
=> Syntax :
		=> select country, count(country)
			from table_name GROUP BY country
		=> select country, count(country)
			*as new_col_result_set* from table_name GROUP BY country


- HAVING : 
=> Syntax :
		=> select country, count(country)
			*as new_col_result_set* from table_name GROUP BY country
				having count(country)>4


# Hands-on Lab : String Patterns, Sorting & Grouping
# (Optional) Hands-on Lab: String Patterns, Sorting and Grouping
# Ungraded External Tool•. Duration: 1 hour1h
# Summary & Highlights
# Practice Quiz
# Practice Quiz•5 questions
# Graded Quiz: Refining Your Results => OKOK

///w3: Functions, Multiple Tables, and Sub-queries///

# Built-in Database Functions

- Most databases comme w/ built-in SQL functions
- Build-in functions can be included as part of SQL statements
- Database functions can significantly reduce the amount of data that needs to be retrieved
- Can speed up data processing

## Aggregate and Column Functions
- INPUT : Collection of values(eg : entire column)
- OUTPUT : Single value
- Examples : SUM(), MIN(), MAX(), AVG(), etc 

## SCALAR and STRING Functions
- SCALAR :  ROUND(), 
- STRING : LENGHT, UCASE, LCASE ...

# Date and Time Built-in Functions
- to extract day, month, year ...
- DATE : YYYYMMDD
- TIME : HHMMSS
- TIMESTAMP (20 digits) : YYYYXXDDHHMMSSZZZZZZ
	=> where Z ... : microseconds
- Date / Time functions : 
 	=> YEAR(), MONTH(), DAY, DAYOFMONTH, DYOFWEEK, DAYOFYEAR, WEEK, HOUR, MINUTE, SECOND, ...

=> Eg : Select DAY(RESCUEDATE) from PETRESCUE 
		where ANIMAL='Cat'
- Date and Time Arithmetic is also possible : CURENT_DATE, CURRENT_TIME
=> Result : YMMDD

# Hands-on Lab: Built-in functions
# (Optional) Hands-on Lab: Built-in functions
# Ungraded External Tool•. Duration: 1 hour1h

# Sub-Queries and Nested Selects
- Sub-Queries : A query inside another query
- Form more powerful queries
- eg : select COLUMN1 from TABLE
		where COLUMN2 = (select MAX(COLUMN2) from TABLE)
- 3 scenarios : 
=> From the table
=> From specific COLUMN (Column Expressions)
=> From clause : Substitute TABLE name w/ a sub-query(Table Expressions)


# Hands-on Lab: Sub-queries and Nested SELECTs 
# (Optional) Hands-on Lab : Sub-queries and Nested SELECTS
# Ungraded External Tool•. Duration: 1 hour1h

# Working with Multiple Tables
- Sub-queries
- Implicit JOIN
- JOIN Operators (INNER JOIN, OUTER JOIN, etc)
	=> CROSS JOIN (also known as Cartesian Join)
	-- syntax : 
	SELECT column_name(s)
	FROM table1, table2;

	=> INNER JOIN
	-- syntax : 
	SELECT column_name(s)
	FROM table1, table2
	WHERE table1.column_name = table2.column_name;


## Sub-queries SYNTAX :  
SELECT * FROM EMPLOYEES
WHERE DEP_ID IN
(SELECT DEPT_ID_DEP FROM DEPARTMENTS);
		WHERE LOC_ID = 'L0002' --> SPECIFIC Value

## Implicit JOIN SYNTAX : 
-- SELECT * FROM EMPLOYEES, DEPARTMENTS; 
=> The result is a full join(or Cartesian join):
	=> Every row in the first table is joined with every row in the 2 one.

=> operations  : retrieving elements/records which matches from 2 tables
-- SELECT * FROM EMPLOYEES, DEPARTMENTS
--		WHERE EMPLOYEES.DEP_ID  = DEPARTMENTS.DEPT_ID_DEP

=> Using Aliases : 
-- SELECT * FROM EMPLOYEES E, DEPARTMENTS D 
-- 		WHERE E.DEP_ID = D.DEPT_ID_DEP; 


# Lab 
# Quiz 

/////////////////////////////////////////////
/// W4 : Accessing Databases using Python ///
/////////////////////////////////////////////

- Python power & efficiency
- Matlab notebook vs Jupyter notebook vs Apache Zeppelin, apache Spark, Databricks cloud .. 
- Python communication w/ DBMS
(Client) => (Notebook) => (API calls) => (DBMS)

## SQL API : 
- Contains functions and operators to communicate w/ DBMS
- requests, errors handling 
## API used by popular SQL-based DBMS systems
		(SQL-based DBMS )		(SQL API)
---------	MySQL		-------- MySQL C API
---------	PostgreSQL	-------- psyscopg2
---------	IBM DB2		-------- ibm_db 
---------	SQL Server	-------- dblib API 
---------	Database acces for MS -------- ODBC
---------	Oracle		-------- OCI
---------	Java		-------- JDBC

# Writing code using DB-API
- Python'q standard API for accessing relational databases (RDB)
- Allows a single program that to work with multiple kinds of RDB 
- Learn DB-API functions once, use them w/ any database

## Applications
- Easy to implement and understand
- Encourages similarity between the Python modules used to access db
- Achieve consistency
- Portable acress databases 
- Broad reach of database connectivity from Python 
			(DB)				(DB API)
---------	IBM Db2		-------- ibm_db 
---------	MySQL		-------- MySQL Connector/Python 
---------	PostgreSQL	-------- psyscopg2
---------	MongoDB		-------- PyMongo

## Concepts of the Python DB API 

- Connection Objects
=> Database connections
=> Manage transactions
- Curso Objects 
=> Database Queries
=> Scroll trough result set 
=> Retrieve results

Connection methods
- cursor, commit, roobacj, close ...
Cursors methods
- callproc, execute, executemany, fetchone ...

[App] = [Cursor object] => [Database]
- cursor sees a db records as list of files ...
- keep the program current position 

## Writing code using DB-API
1. import the dbmodule import connect 
2. create connection object
3. create a cursor object
4. run queries
5. free resources (Cursor.close, connection.close)

# Connecting to a database using ibm_db API
- provides a variety of useful Python functions for acessing and manipulating data an IBM data server Database 
- uses IBM Data Server Driver for ODBC and CLI APIs to connect to IBM DB2 and Informix 

# Lab: Create Database Credentials

# Hands-on Lab: Connecting to a database instance
# Ungraded External Tool => OK

# Creating tables, loading data and querying data
## Creating tables
- using the python db api 
- ibm_db.exec_immediate(conn_obj, "sql_queries") =>
## Insert data into the table
- ibm_db.exec_immediate(conn_obj, "sql_queries") =>
## Querying data
- ibm_db.fetching(stmt)
- Using pandas
	=> read_sql("sql_queries", conn_obj )

# Lab

# Introducing SQL Magic

# Hands-on Tutorial: Accessing Databases with SQL magic
# Ungraded External Tool•. Duration: 20 minutes20 min
# (Optional) Hands-on Tutorial: Accessing Databases with SQL magic
# Ungraded External Tool

# Analyzing data with Python

# Lab 
# Quiz 

///W5: Assignment Preparation: Working with real-world data sets and built-in SQL functions///

# Working with Real World Datasets
- Working with CSV files (.CSV : COMMA SEPARATED VALUES)
- the first corresponds of the name of columns  
- Data is non-sensitive (low != upper case) => SOLUTION "Id" == ID
- Spaces => Under_cores
- \ backslash are great to scape characters and go to the next line in case of long queries

# Getting Table and Column Details
- Getting the List of tables in the database 
=> DB2 : SYSCAT.TABLES
=> SQL Server : INFORMATION_SCHEMA.TABLES
=> Oracle : ALL_TABLES or USER_TABLES

- syntax : 
 SELECT * FROM SYSCAT.TABLES 
 or
 SELECT TABSCHEMA, TABNAME, CREATE_TIME
	FROM SYSCAT.TABLES
	WHERE TABSCHEMA = 'DB2_USERNAME'

## Getting a list of columns in the database
SELECT * FROM SYSCAT.COLUMNS
	WHERE TABNAME = 'TABLE_NAME'

or 
SELECT DISTINCT(NAME), COLTYPE, LENGHT
	FROM SYSIBM.SYSCOLUMNS
	WHERE TBNAME = 'TABLE_NAME'

# LOADing Data => OK

/////////////////////////////////////////////
/// W5 : Accessing Databases using Python ///
/////////////////////////////////////////////
Assignment => OK
# Course Wrapup 

This course is part of:
-IBM Data Analyst Professional Certificate
-Applied Data Science Specialization
-IBM Data Science Professional Certificate
-IBM Data Engineering Foundations Specialization
-IBM Data Engineering Professional Certificate
 (Coming in the Future)

As a next step, you can explore other courses in these programs, starting with: 
-Data Engineering and Data Analytics tracks: Data Analysis with Python
-Data Engineering track: NoSQL Fundamentals (Coming soon...)

/////////////////////////////////////////////
/// W6 : Advanced SQL For DataScience Engineer///
/////////////////////////////////////////////

/// Views, stored procedures and transactions ///
# Views
# Stored Procedures
# ACID Transactions

/// Joins statements ///
# Join Overview
# Inner Join
# Outer Joins

///Lab///
///Quiz///

## Module 2: Introduction to Relational Databases and Tables
## Module 3: Intermediate SQL
## Module 4: Accessing Databases using Python
## Module 5: Course Assignment
## Module 6: Bonus Module: Advanced SQL for Data Engineers (Honors)
## Lab

Notebooks: 

- [All Notebooks](./lab/notebooks/)

## References


