![wallpaper](https://user-images.githubusercontent.com/67295703/162627938-4eb70598-3de7-4854-8d17-11dd16fe27bb.jpg)
# Car Market in Poland
Exploratory data analysis and car prices prediction. 

[Link to the full notebook](https://nbviewer.org/github/CyperStone/car-market-poland/blob/main/car_market_in_poland.ipynb)

## Project Overview
* Analyzed and visualized the car market in Poland (e.g the most popular brands and models, frequent features, or price distribution)
* Created a model that predicts car prices (RMSE ~ 5250 USD) to help people to make a initial estimate of vehicle price bracket
![alt text](https://github.com/CyperStone/car-market-poland/blob/main/visualization/predicted_vs_actual.png)

## Technologies and Resources
* Python Version: 3.9
* Packeges: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly, currency_converter, ast, pickle
* Dataset: https://www.kaggle.com/datasets/bartoszpieniak/poland-cars-for-sale-dataset

## About the Dataset
The dataset contains 208,304 observations with 25 variables, posted from 2021-03-26 to 2021-05-05 on the polish website.

Variables description:
* ID - unique offer's ID
* Price - car price
* Currency - currency (PLN or EUR)
* Condition - new or used
* Vehicle_brand - brand of car
* Vehicle_model - model of car
* Vehicle_generation - model generation of car
* Vehicle_version - version of car
* Production_year - year of car production
* Mileage_km - total distance that the car has driven in km
* Power_HP - car engine power in HP
* Displacement_cm3 - car engine size in cubic centimeters
* Fuel_type - car fuel type
* CO2_emissions - car CO2 emissions in g/km
* Drive - type of car drive
* Transmission - type of car transmission
* Type - car body style
* Doors_number - number of car doors
* Colour - car body color
* Origin_country - origin country of the car
* First_owner - whether the owner is the first owner
* First_registration_date - date of first registration
* Offer_publication_date - date of offer publication
* Offer_location - address provided by the issuer
* Features - string with list of additional car features (ABS, electric windows etc.)

## Technical Overview
* **Data Preprocessing**:
  * Removed duplicated rows and columns with more than 30% of missing values
  * Extracted province names from whole offer location addresses
  * Used CurrencyConverter to convert prices to USD respecting exchange rate on offer publication date
  * Extracted day and month and day of the week from offer publication date
  * Created custom transformer for converting the least frequent categorical feature values to one common value
  * Created custom transformer for extracting additional car features
  * Created nested pipeline to perform scaling, encoding, and imputing missing values differently for various columns
* **Model Building**:
  * Chose RMSE metric to penalise large errors
  * Tried different models: Ridge Regression, LinearSVR, DecisionTreeRegressor and XGBRegressor
  * Selected XGBRegressor since it outperformed other models
  * Tuned XGBRegressor hyperparameters
  * Tested final model on test set

## EDA Highlights
* **Distribution of car prices on the Polish market**:
![alt text](https://github.com/CyperStone/car-market-poland/blob/main/visualization/prices.png)
* **Distribution of car production years on the Polish market**:
![alt text](https://github.com/CyperStone/car-market-poland/blob/main/visualization/production_years.png)
* **The most popular car brands on the Polish market**:
![alt text](https://github.com/CyperStone/car-market-poland/blob/main/visualization/brands.png)

## Final Model
* As mentioned in the introduction, the final model achieved RMSE ~ 5250 USD and R2 score = 0.944
* Car features that were most valuable for predictive modeling:
![alt text](https://github.com/CyperStone/car-market-poland/blob/main/visualization/feature_importances.png)
* Prediction errors distribution of the final model:
![alt text](https://github.com/CyperStone/car-market-poland/blob/main/visualization/errors_histogram.png)
