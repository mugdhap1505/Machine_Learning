# Machine Learning
 
 This repository consists of implementation of basic Machine Learning algorithms. 

# Linear Regression
## Regression Model to Predict Bike Rentals in Washington, DC

## Problem Description:
The bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions, precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors. The core data set for this project is related to the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare System, Washington D.C.

## Dataset Description:
The data contains conditions (11 xi’s) on a daily basis along with the count of bikes that were rented each day (y).
- season : season (1:springer, 2:summer, 3:fall, 4:winter)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- holiday : whether a day is a holiday or not (0=no, 1 = yes)
- weekday : day of the week (0-6)
- workingday : if day is neither a weekend nor holiday, = 1, otherwise = 0.

- weathersit :

 1: Clear, Few clouds, Partly cloudy, Partly cloudy

 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

 4: Heavy Rain + Ice Pellets + Thunderstorm + Mist, Snow + Fog

- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- cnt: count of total rental bikes including both casual and registered

## Aim:
To develop a regression model that will predict how many bikes will be rented on any particular day based on these 11 conditions.

## Model development.
Should randomize the data file and put 439 data sets in your training set file, 146 data sets in your cross validation set data file and 146 data sets in your test set data file.

Dataset: BikeAll.txt

# Classification with Logistic Regression

## Problem description: 
Apply Logistic Regression to the Divorce Predictors Scale to classify whether married couples will eventually divorce. 
Each of the questions in the appendix were answered on a scale of 0 (Strongly Disagree) to 4 (Strongly Agree). Our data set has 54 questions (features, n) and 170 sets of data (m). 

## Dataset Description:
Data file is formatted with the first line containing m and n, tab separated). Then each line following has 55 entries, the first 54 are the responses to the questions below (0-4) followed by a y coded as either 0 (divorce class) or 1 (did not divorce class). 

## Aim:
To develop a binary classifier to predict whether a person will divorce (0) or stay married (1).

Dataset used: DivorseAll.txt

Training file: Patil_Mugdha_Train.txt

Testing file: Patil_Mugdha_Test.txt

Code file: Patil_Mugdha_P2.py

# Building a Spam Filter using a Naïve Bayes Classifier

## Problem description and Dataset:
 
To build a Naïve Bayes Spam filter. A labeled training set file and a labeled test set file is provoided. Both files will have the same format. Each line will start with either a 1 (Spam) or a 0 (Ham), then a space, followed by an email subject line. A third file will contain a list of Stop Words—common words that you should remove from your vocabulary list. Format of the Stop Word list will be one word per line.

## Aim:
the program should prompt the user for the name of a training set file in the format described above and for the name of the file of Stop Words. It should create a vocabulary of words found in the subject lines of the training set associated with an estimated probability of each word appearing in a Spam and the estimated probability of each word appearing in a Ham email. The program should then prompt the user for a labeled test set and predict the class (1 = Spam, 0 = Ham) of each subject line using a Naïve Bayes approach

Training file: SHTrain.txt

Testing file: SHTest.txt

Stopwords file: StopWords.txt

Code file: Patil_Mugdha_P3.py

Jupyter notebook: https://github.com/mugdhap1505/Machine_Learning/blob/master/Naive%20Bayes%20Implementation.ipynb

## K-means Clustering

## Problem and Dataset Description
Aim is to implement a K-means clustering algorithm.The data file will be formatted with first line containing m and n, tab separated, wherem is the number of lines of data and n is the number of features (for this assignment n will be 2 but assume we still put it into the file.)

## Aim:
1. The program should prompt the user for the name of a data file.

2. Prompt the user for the name of a file containing two initial centroids.

3. Print out the coordinates of the two initial centroids.

4. Print out a plot of the data to the screen, including the two initial centroids.

5. Run K means (K=2) to cluster the data into two groups.

6. Print out a plot of the cluster data with each cluster color coded along with the
final centroids.

7. Print out the coordinates of the final centroids.

8. Compute and print out the overall error (J function presented in class) for the
entire data set.

Dataset file: P4Data.txt

Code file: Patil_Mugdha_P4.py

Jupyter notebook: https://github.com/mugdhap1505/Machine_Learning/blob/master/K-means%20.ipynb
