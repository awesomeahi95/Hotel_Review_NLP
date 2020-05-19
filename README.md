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

The initial shape of the dataset was (35078,5). The 5 columns was as expected, but there were double the number of rows as the number of reviews scraped. There were null rows with only hotel_name and no other values, so I removed those rows, bringing us back to the expected 17538.

This project entailed the use of classification models, and for reliable results, I had to remove reviews to undo class imbalance. Using this visualisation I saw that were much less reviews with a score of 1 compared to reviews with a score of 3, 4, and 5. To combat this imbalance, I randomly removed reviews with scores of 2, 3, 4, and 5, to match with 1 (1881 reviews). 

<h5 align="center">Histogram of Scores for All Hotels (With  Class Imbalance (Left) vs Without  Class Imbalance (Right))</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/with_class_imbalance.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/without_class_imbalance.png' width=500></td></tr></table>

I combined the review p1 and review p2 column into one to make future vectorisation much easier, then I saved the cleaned dataset as a csv, for the next stage.

<a name="Further_EDA_and_Preprocessing"></a>
### Further EDA and Preprocessing

The cleaned dataset had a shape of (9405,4). I started with some analysis on the text columns; review and review summary.

Using the FreqDist function in the ntlk library I plotted a graph with the most frequent words and phrases in both columns. Stopwords were removed to capture the more meaningful words.

<h5 align="center">Distribution Plot of Frequent Words and Phrases in Text (Review Summary (Left) and Review (Right))</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/freq_dist_review_sum.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/freq_dist_review.png' width=500></td></tr></table>

I had noticed a lot of the most frequent words in the review text happened to be words with no sentimental impact, so I iteratively removed unmeaningful words such as 'room', 'hotel', 'hilton' etc. I did this as a precaution, as some of these words may impact my model accuracies.

<h5 align="center">World Cloud of Frequent Words and Phrases in Text After Removing Unmeaningful Words(Review Summary (Left) and Review (Right))</h5>
<table><tr><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/word_cloud_review_sum.png' width=500></td><td><img src='https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/word_cloud_review.png' width=500></td></tr></table>

To narrow down the feature words I applied stemmation and lemmitisation to both the reviews and review summaries. 

<h5 align="center">Example of Lemmatisation and Stemmation Applied to a Review and Review Summary</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/lemm_stemm_ex.png" width=800>
</p>

Stemmation had broken down some words into words that don't exist, whereas lemmitisation had simplified adjectives and verbs to their root form. I chose to continue with the lemmitised version of the texts for further processing.

Prior to vectorising the current dataset, I did a train, test split to save the test data for after modelling.

Using the lemmed texts for review and review summary I used TF-IDF vectorisation, leaving me with a vectorised dataset with 234 words and phrases (178 from reviews and 56 from review summaries).


<a name="Modelling"></a>
### Modelling:



<a name="Evaluation"></a>
### Evaluation


