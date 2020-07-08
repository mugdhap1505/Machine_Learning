# Machine Learning

1. Visualization of three types of flowers in the Iris dataset using Matplotlib

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

Dataset: 
Code file: 

3. Logistic Regression

4. Naive Bayes Implemtation

5. K-means Implementation
