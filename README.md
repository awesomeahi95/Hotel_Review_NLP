# Hilton London

In the modern day, public discussion and critiquing of products and services occurs beyond dedicated mediums, and now also takes place in the realm of social media, too.

Potential customers, could have their hotel choice be influenced by a tweet. Opinions are shared constantly on social media platforms, and are read by their followers. The knowledge, of what these followers think about our hotel, from reading these online posts, could help us better understand the general public's perception of our hotel.

By using sentiment analysis, on existing hotel reviews from Tripadvisor.com, I created a model that can quantify on a scale of 1-5, how the author of a tweet on twitter, or a post on a reddit thread, feels about our hotel, and as a result, also how the readers think about us.

## Table of Contents

1. [ File Descriptions ](#File_Description)
2. [ Strucuture ](#Structure)
3. [ Executive Summary ](#Executive_Summary)
   * [ Webscraping ](#Webscraping)
   * [ 1. Early EDA and Cleaning ](#Early_EDA_and_Cleaning)
   * [ 2. Further EDA and Preprocessing ](#Further_EDA_and_Preprocessing) 
   * [ 3. Modelling ](#Modelling)
   * [ 4. Evaluation ](#Evaluation)

<a name="File_Description"></a>
## File Descriptions

- Tripadvisor_Webscrape: folder containing all webscraping files
    - Tripadvisor: folder containing .py files and spiders used
        - spiders: folder containing spider files and datasets
            - hotels.py: main spider .py file for scraping hotel reviews from Tripadvisor
            - tripadvisor_scraped_hotel_reviews.csv: csv file with data to be used for project
        - _init_.py, items.py, middlewares.py, pipelines.py, settings.py: default scrapy files used for webscrape
    - scrapy.cfg: scrap config file
- 1.Early_EDA_and_Cleaning.ipynb: notebook with early data exploration and data manipulation
- 2.Further_EDA_and_Preprocessing.ipynb: notebook with feature engineering and nlp preprocessing
- 3.Modelling.ipynb: notebook with all the models created
- Classification.py: contains classes for classifcation methods
- 1.tripadvisor_scraped_hotel_reviews.csv: webscraped data before any changes
- 2.hotel_reviews_structured.csv: data after balancing and cleaning
- 3.x_train_data.csv: training data with x values from preprocessed dataset
- 3.y_train_data.csv: training data with y values from preprocessed dataset
- final_model.pkl: final model saved using pickle.
- Hilton_London.pdf: presentation summarising project process and findings


<a name="Structure"></a>
## Structure of Notebooks:
1. Early EDA and Cleaning
   - 1.1 Imports
   - 1.2 Checking for Nulls
   - 1.3 Converting Score Column
   - 1.4 Adjusting Class Imbalance for Scores
   - 1.5 Joining Review Part 1 with Review Part 2 in New Column Review
   - 1.6 Removing Review Part 1 and Review Part 2 Columns
   - 1.7 Saving Structured Dataset as a CSV

2. Further EDA and Preprocessing
   - 2.1 Imports
   - 2.2 Checking Frequency of Words and Phrases in Review Summaries
   - 2.3 Flattening Reviews to Check Word Frequency
   - 2.4 Checking Frequency of Words and Phrases in Reviews
   - 2.5 Stemming and Lemming
   - 2.6 Train Test Split
   - 2.7 TF-IDF Vectorisation for Reviews
   - 2.8 TF-IDF Vectorisation for Review Summaries
   - 2.9 Joining Reviews With Review Summaries
   - 2.10 Saving Preprocessed Dataset as a CSVs

3. Modelling
   - 3.1 Imports
   - 3.2 Train and Validation Split
   - 3.3 Decision Tree (Baseline)
   - 3.4 Random Forest
   - 3.5 Logistic Regression
   - 3.6 Support Vector Machines
   - 3.7 Guassian Naive Bayes

4. Evaluation
   - 4.1 Imports
   
   
<a name="Executive_Summary"></a>
## Executive Summary

As a struggling US life insurance company, our goal is to increase revenues by 2% using a risk premium based on premature deaths per state. We aim to hike insurance contract prices across the states that pause the highest future risk of premature death rates. This should in turn take our net premium growth rate above annual inflation which we have been on par with for the last five years and move to a more risk-adjusted business model which is key in our industry. 


<a name="Webscraping"></a>
### Webscraping

I set a goal of a minimum of 5000 reviews to scrape, before choosing the specific hotels. I then chose the 5 Hilton hotels with the highest number of reviews, to scrape; London Gatwick Airport, London Metropole, London Euston, London Croydon, and London - West End. 
Between these 5 hotels there were 17538 reviews, I had plenty room to filter or drop reviews and retain at least my minimum of 5000.

<h5 align="center">Tripadvisor Review Example</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/Tripadvisor_Review_Example.png" width=600>
</p>

The structure of each review consisted of a 1-5 scale score rating in bubble form, a review summary, and a detailed review split into p1 and p2 (depending on if there was a read more option). Each page on tripadvisor had 5 reviews per page, so I had to navigate between pages using tripadvisor's next page function. 

The root URL I used was 'www.tripadvisor.co.uk'

The 5 starting URL extensions I used were:
- '/Hotel_Review-g187051-d239658-Reviews-Hotel_Hilton_London_Gatwick_Airport-Crawley_West_Sussex_England.html/'
- '/Hotel_Review-g186338-d193089-Reviews-Hilton_London_Metropole-London_England.html/'
- '/Hotel_Review-g186338-d192048-Reviews-Hilton_London_Euston-London_England.html/'
- '/Hotel_Review-g186338-d193102-Reviews-DoubleTree_by_Hilton_Hotel_London_West_End-London_England.html/'
- '/Hotel_Review-g504167-d192599-Reviews-Hilton_London_Croydon-Croydon_Greater_London_England.html'

From these pages I chose to extract 5 different features:
- hotel_name
- review_summary
- review_p1
- review_p2
- score

I used a scrapy spider to crawl the website to scrape the requested data. Scrapy proved the be efficient and fast at extracting the data. I ran the spider script (hotels.py) for around 20 minutes, on the 13th May 2020.

<h5 align="center">Histogram of Scores for Each Hotel</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/hotels_and_score.png" width=600>
</p>


<a name="Early_EDA_and_Cleaning"></a>
### Early EDA and Cleaning: 

The initial shape of the csv was (35078,5). The 5 columns was as expected, but there were double the number of rows as the number of reviews scraped. There were null rows with only hotel_name and no other values, so I removed those rows, bringing us back to the expected 17538.

This project entailed the use of classification models, and for reliable results, I had to remove reviews to undo class imbalance. Using this visualisation I saw that were much less reviews with a score of 1 compared to reviews with a score of 3, 4, and 5. To combat this imbalance, I randomly removed reviews with scores of 2, 3, 4, and 5, to match with 1 (1881 reviews). 

<h5 align="center">Histogram of Scores for All Hotels (With  Class Imbalance vs Without  Class Imbalance)</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/with_class_imbalance.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/without_class_imbalance.png' width=500></td></tr></table>



<a name="Further_EDA_and_Preprocessing"></a>
### Further EDA and Preprocessing

The initial step included setting up a structured framework to the data. We initially setup a train-test split at a 2/3 // 1/3 respective split to include at least 1000 observations in our testing test. For the regularisation models we used StandardScaler to standardise our data.

<a name="Modelling"></a>
### Modelling:

We then performed 5-fold cross validation across our models for our training data to validate the completeness and quality of our data-model interaction. This also allowed us to select the highest performing models within each model group/type. 
The initial baseline model was a multivariate linear regression using the Ordinary Least Squares (OLS) method. With 5 different variables the model yielded an R^2 score of 0.5223 which was an encouraging sign as an first instance model.

We then proceeded to perform a polynomial transformation to our variables to factor interactions of independent variables and their explanatory power of premature deaths. We ran three levels of polynomial regression: quadratic, qubic, quartic and quintic. The best results came out of the quadratic transformation with an improved R^2 score of 0.5554. We elected to keep this transformation.  

Using a z-score standard scaler method we took our poly-transformed data and scaled it before applying the regularisation methods. 

To improve our coefficients and simplify our model we decided to run a series of regularisation technqiues using lasso, ridge and elastic net regressions with several alpha levels. In total we had 60 models with regularisation (20 Lasso, 20 Ridge, and 20 Elastic Net). We once again applied the 5-fold validation process to find the best R^2 value across model types with the optimal level.

<h5 align="center">Table Comparing Best Models</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/LaVie-Insurance/blob/master/Images/Best_Models_Table.png" width=850>
</p>

<h5 align="center">Bar Chart Comparing Best Models</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/LaVie-Insurance/blob/master/Images/Best_Models_BarChart.png" width=850>
</p>

Our rationale at this point is to come back to our stakeholders and make sure the regression is as simple as possible to apply it in a business context. We selected a lasso model yielding an R^2 of 0.5241 on the training set with an alpha level of 47.373684. 

Finally, we ran the model on the test set and got an R^2 of 0.5626 illustrating some accuracy in our model and a possibility to deploy it for our business application. 

<h5 align="center">Final Model Coefficients</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/LaVie-Insurance/blob/master/Images/Best_Model_Coefs.png" width=450>
</p>

<h5 align="center">Final Model Residuals Plot</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/LaVie-Insurance/blob/master/Images/Residuals_of_best_model.png" width=450>
</p>


<a name="Evaluation"></a>
### Evaluation

We believe our final model provides us enough prediction accuracy to implement the life insurance premium increase. We are aiming to perform this increase as a function of YPLL predictions scaled to the population size. This model should allow us not only to increase our revenues but to shift towards a more risk-adjusted revenue approach which we can replicate in the future. 
