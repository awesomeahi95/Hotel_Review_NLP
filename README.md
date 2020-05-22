<h1 align='center'> HILTON HOTEL LONDON REVIEWS CLASSIFIER </h1>

<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/five_stars.png" width=600>
</p>

In the modern day, public discussion and critiquing of products and services occurs beyond dedicated mediums, and now also takes place in the realm of social media, too.

Potential customers, could have their hotel choice be influenced by a tweet. Opinions are shared constantly on social media platforms, and are read by their followers. The knowledge, of what these followers think about our hotel, from reading these online posts, could help us better understand the general public's perception of our hotel.

By using sentiment analysis, on existing hotel reviews from Tripadvisor.com, I created a model that can quantify on a scale of 1-5, how the author of a tweet on twitter, or a post on a reddit thread, feels about our hotel, and as a result, also how the readers think about us.

Email: candyahs@gmail.com <br>
LinkedIn: www.linkedin.com/in/ahilan-srivishnumohan/ <br>
Medium: www.medium.com/@candyahs <br>

## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>

1. [ File Descriptions ](#File_Description)
2. [ Structure ](#Structure)
3. [ Executive Summary ](#Executive_Summary)
   * [ 1. Webscraping, Early EDA, and Cleaning ](#Webscraping_Early_EDA_and_Cleaning)
       * [ Webscraping ](#Webscraping)
       * [ Early EDA and Cleaning](#Early_EDA_and_Cleaning)
   * [ 2. Further EDA and Preprocessing ](#Further_EDA_and_Preprocessing) 
   * [ 3. Modelling and Hyperparameter Tuning ](#Modelling)
   * [ 4. Evaluation ](#Evaluation)
       * [ Future Improvements ](#Future_Improvements)
</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Data</strong>: folder containing all data files
    * <strong>1.tripadvisor_scraped_hotel_reviews.csv</strong>: webscraped data before any changes
    * <strong>2.hotel_reviews_structured.csv</strong>: data after balancing and cleaning
    * <strong>3.x_train_data.csv</strong>: training data with x values from preprocessed dataset
    * <strong>3.y_train_data.csv</strong>: training data with y values from preprocessed dataset
    * <strong>4.x_test_data.csv</strong>: test data with x values from preprocessed dataset
    * <strong>4.y_test_data.csv</strong>: test data with y values from preprocessed dataset
* <strong>Images</strong>: folder containing images used for README and presentation pdf
* <strong>Models</strong>: folder containing trained models saved with pickle
    * <strong>Adabooost.pkl, Decision Tree.pkl, KNN.pkl, Logistic Regression.pkl, Naive Bayes.pkl, Random Forest.pkl, Stacking.pkl, SVM.pkl, Voting.pkl, XGBoost.pkl</strong>
* <strong>Tripadvisor_Webscrape</strong>: folder containing all webscraping files
    * <strong>Tripadvisor</strong>: folder containing .py files and spiders used
        * <strong>spiders</strong>: folder containing spider files and datasets
            * <strong>hotels.py</strong>: main spider .py file for scraping hotel reviews from Tripadvisor
            * <strong>tripadvisor_scraped_hotel_reviews.csv</strong>: csv file with data to be used for project
        * <strong>_init_.py, items.py, middlewares.py, pipelines.py, settings.py: default scrapy files used for webscrape</strong>
    * <strong>scrapy.cfg</strong>: scrap config file
* <strong>1.Webscraping_Early_EDA_and_Cleaning.ipynb</strong>: notebook with early data exploration and data manipulation
* <strong>2.Further_EDA_and_Preprocessing.ipynb</strong>: notebook with feature engineering and nlp preprocessing
* <strong>3.Modelling_and_Hyperparameter_Tuning.ipynb</strong>: notebook with all the models created
* <strong>4.Evaluation</strong>: notebook with final model selection, testing, and model evaluation
* <strong>Classification.py</strong>: contains classe with classifcation methods
* <strong>Ensemble.py</strong>: contains classe with ensemble methods
* <strong>Hilton__Hotel_Presentation.pdf</strong>: presentation summarising project case, processese, and findings
</details>

## Structure of Notebooks:
<details>
<a name="Structure"></a>
<summary>Show/Hide</summary>
<br>
1. Early EDA and Cleaning
   * 1.1 Webscrape Process Explained
   * 1.2 Imports
   * 1.3 Checking for Nulls
   * 1.4 Converting Score Column
   * 1.5 Adjusting Class Imbalance for Scores
   * 1.6 Joining Review Part 1 with Review Part 2 in New Column Review
   * 1.7 Removing Review Part 1 and Review Part 2 Columns
   * 1.8 Saving Structured Dataset as a CSV

2. Further EDA and Preprocessing
   * 2.1 Imports
   * 2.2 Checking Frequency of Words and Phrases in Review Summaries
   * 2.3 Checking Frequency of Words and Phrases in Reviews
   * 2.4 Stemming and Lemming
   * 2.5 Train Test Split
   * 2.6 TF-IDF Vectorisation for Reviews
   * 2.7 TF-IDF Vectorisation for Review Summaries
   * 2.8 Joining Reviews With Review Summaries
   * 2.9 Saving Preprocessed Dataset as CSVs

3. Modelling and Hyperparameter Tuning
   * 3.1 Imports
   * 3.2 Train and Validation Split
   * 3.3 Decision Tree (Baseline)
   * 3.4 Random Forest
   * 3.5 Logistic Regression
   * 3.6 Support Vector Machines
   * 3.7 Guassian Naive Bayes
   * 3.8 KNN
   * 3.9 Adaboost (Logistic Regression)
   * 3.10 XGBoost (Logistic Regression)
   * 3.11 Voting
   * 3.12 Stacking
   * 3.13 All Models Compared
   * 3.14 Best Model (Logistic Regression) - Deeper Look
   * 3.15 Saving Best Model

4. Evaluation
   * 4.1 Imports
   * 4.2 Best Model Selection
   * 4.3 Best Model Tested
   * 4.4 Deeper Diver Into Best Model
   * 4.5 Application Deployability
</details>  
   
<a name="Executive_Summary"></a>
## Executive Summary


<a name="Webscraping_Early_EDA_and_Cleaning"></a>
### Webscraping, Early EDA, and Cleaning:
<details open>
<summary>Show/Hide</summary>
<br>

<a name="Webscraping"></a>
#### Webscraping

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
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/histogram_of_scores_for_each_hotel.png" width=600>
</p>


<a name="Early_EDA_and_Cleaning"></a>
#### Early EDA and Cleaning: 

The initial shape of the dataset was (35078,5). The 5 columns was as expected, but there were double the number of rows as the number of reviews scraped. There were null rows with only hotel_name and no other values, so I removed those rows, bringing us back to the expected 17538.

This project entailed the use of classification models, and for reliable results, I had to remove reviews to undo class imbalance. Using this visualisation I saw that were much less reviews with a score of 1 compared to reviews with a score of 3, 4, and 5. To combat this imbalance, I randomly removed reviews with scores of 2, 3, 4, and 5, to match with 1 (1881 reviews). 

<h5 align="center">Histogram of Scores for All Hotels (With  Class Imbalance (Left) vs Without  Class Imbalance (Right))</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/histogram_of_scores_for_all_hotels.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/histogram_of_scores_for_all_hotels_after_balancing.png' width=500></td></tr></table>

I combined the review p1 and review p2 column into one to make future vectorisation much easier, then I saved the cleaned dataset as a csv, for the next stage.
</details>  

<a name="Further_EDA_and_Preprocessing"></a>
### Further EDA and Preprocessing

The cleaned dataset had a shape of (9405,4). I started with some analysis on the text columns; review and review summary.

Using the FreqDist function in the ntlk library I plotted a graph with the most frequent words and phrases in both columns. Stopwords were removed to capture the more meaningful words.

<h5 align="center">Distribution Plot of Frequent Words and Phrases in Text ( Review Summary (Left) and Review (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/freq_dist_review_sum.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/freq_dist_review.png' width=500></td></tr></table>

I had noticed a lot of the most frequent words in the review text happened to be words with no sentimental impact, so I iteratively removed unmeaningful words such as 'room', 'hotel', 'hilton' etc. I did this as a precaution, as some of these words may impact my model accuracies.

<h5 align="center">World Cloud of Frequent Words and Phrases in Text After Removing Unmeaningful Words ( Review Summary (Left) and Review (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/word_cloud_review_sum.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/word_cloud_review.png' width=500></td></tr></table>

To narrow down the feature words I applied stemmation and lemmitisation to both the reviews and review summaries. 

<h5 align="center">Example of Lemmatisation and Stemmation Applied to a Review and Review Summary</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/lemm_stemm_ex.png" width=600>
</p>

Stemmation had broken down some words into words that don't exist, whereas lemmitisation had simplified adjectives and verbs to their root form. I chose to continue with the lemmitised version of the texts for further processing.

Prior to vectorising the current dataset, I did a train, test split to save the test data for after modelling.

Using the lemmed texts for review and review summary I used TF-IDF vectorisation with an ngram range of 2, leaving me with a vectorised dataset with 138 words and phrases (112 from reviews and 26 from review summaries). I then saved the x and y train data in separate csv files for modelling.


<a name="Modelling"></a>
### Modelling:

I have created .py files; Classifiction.py and Ensemble.py with classes, that contain functions to simplify the modelling process, and to neaten up the modelling notebook.

I did another data split into Train and Validation data in preparation for using GridSearch Cross Validation. I also chose Stratified 5-fold has a my choice for cross validating.

For the majority of models I created, I applied hyperparameter tuning, where I started with a broad range of hyperparameters, and tuned for optimal train accuracy and validation accuracy. 


<h5 align="center">Table Comparing Best Models</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/all_models.png" width=600>
</p>

Initially, I thought the validation accuracy was low for most of the models I created, but when considering these models were attempting to classify for 5 different classes, 0.45 and greater seems very reasonable (where 0.2 = randomly guessing correctly).

I have saved all the models using the pickle library's dump function and stored them in the Models folder.


<a name="Evaluation"></a>
### Evaluation

I focused on 3 factors of defining a good model:

1. Good Validation Accuracy
2. Good Training Accuracy
3. Small Difference between Training and Validation Accuracy

I chose the Stacking ensemble model ( (Adaboost with log_reg_2) stacked with log_reg_2 ) as my best model, because it has the highest validation accuracy with only around 3.5% drop from train to validation in accuracy. I wanted to minimise overfitting and make the model as reusable as possible. Stacking achieved a reasonable training accuracy as well, although it did not reach the level of some of the other ensemble techniques.

I next tested the best model with the earlier saved test data. The model managed to get a high test accuracy, similar to the validation data from the model training stage. This is very good, proving that prioritising a high validation score, and minimising the difference between train and validation accuracy, has helped it classify new review texts very well.

<h5 align="center">Test Results</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/test_results.png" width=600>
</p>

Looking at the precision, recall, and f1 score, I also noticed the scores were higher around scores of 1 and 5, lower for 2, 3, and 4. This shows that the models performs well on more extreme opinions on reviews than mixed opinions.

Looking into different metrics and deeper into my best model; Stacking, I learnt that most the False Postives came from close misses (e.g. predicting a score of 4 for a true score of 5). This is best shown by these two confusion matrixes (validation and test). 

<h5 align="center">Confusion Matrix for Validation and Test Data Predictions ( Validation (Left) and Test (Right) )</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/validation_conf_matrix.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/test_conf_matrix.png' width=500></td></tr></table>

The adjacent squares of the diagonal going across the confusion matrix, shows that the model's second highest prediction for a given class (review score) is always a review score that is +-1 the true score.
Very few reviews that have a score of 5, have been predicted to have a score of 1 or 2. This is very relieving to know, the majority of the error for the model, is no different to the error a human may make classifying a review to a score with a scale of 1-5.

- most errors were near misses (e.g. 5 predicted as 4)
- extreme scores (1 and 5) were relatively accurate
- comparable to human prediction
- reusable and consistent


Given the classifcation problem is 5 way multi-class one and the adjacent classes can have overlap in the english language even to humans, this model I have created can be deployed.

Applying this model will address the problem of not having a full understanding of public opinion of our hotel. We can apply this to new sources for opinions on our hotel and yield more feedback then we did had before.

<a name="Future_Improvements"></a>
### Future Improvements

- Model using RNN and LSTMs - see if better accuracy can be achieved
- Create a working application to test new reviews written by people
- Try a different pre-processing approach and see if model performances change
- Bring in new sources of data to see if there are significant differences on frequent words used