# Sentiment_Analysis_Squid_Game

**Overview:**

With the recent advances in deep learning, the ability of algorithms to analyse text has improved considerably. Advanced artificial intelligence can be used as an effective tool for in-depth research on test analysis. It is very important to classify the incoming customer reviews about a particular brand on key aspects of the brand’s product/service that customers care and the Customers’ underlying intensions and reactions. These basic ideas when combined becomes an important tool which can be used for analysing thousands of conversations related to brand with human level accuracy. 

Sentiment analysis is the most common text classification tool. It does the analysis on the incoming message and identifies whether the underlying sentiment is positive, negative, or neutral. Today, in all domains of social activity there is a presence social media. Sharing information about the product, event, service, or place is one of the common activities in the social media. 

Opinion extraction has become an important area of research in the field of Natural Language processing(NLP) as people express their opinion on social media sites and on commercial sites. One of the neural network architectures used in sentiment analysis is RNN(recurrent neural networks). RNN are good at processing sequence data for predictions. Hence, they are very useful for deep learning applications like speech recognition, speech synthesis, natural language understanding, etc.

**Problem Statement:**

Netflix has been investing in foreign language programming since 2015. It has spent more than $1 billion on Korean programs alone. Squid game is the first Korean show to break through on this scale, and it is driving millions of new viewers to other East Asian series. It will be globalizing the entertainment business, creating a platform for people from more than 190 countries to watch stories from all over the world. To be a global leader in entertainment business, Netflix can make use of sentiment analysis using Long Short-Term Memory (LSTM), Simple Recurrent Neural Network (SimpleRNN), Gated Recurrent Units(GRU) and Bidirectional LSTM models to develop better insights about customer preferences thereby gaining competitive advantage and increase their brand value.

**Business Case:**

Streaming businesses can’t always predict what consumers will want until they have a consumer’s attention, which is why data is essential to win that precious engagement. Such data driven decisions need insights from multiple sources like viewership data to drive content discovery, engagement, attention, and ultimately revenue. In this project, we tried to understand impact of social media and word of mouth for streaming content discovery. Netflix also certainly used social data to inform their content recommendation systems and surface the series in the feeds of new audiences around the world, tying back to the need for personalization algorithms

Squid Game was not heavily marketed in the United States. In fact, aside from an initial trailer that dropped on Netflix’s YouTube channel September 1, it took more than a week after the show’s premiere to generate buzz within popular social accounts before taking off like a rocket. Here are a few key moments over the first 30 days of launch. Social media played a vital role in the success of Squid Game.
 

**Data Source:**

The data for this application was obtained from the Kaggle.com website. The primary information came from: https://www.kaggle.com/deepcontractor/squid-game-netflix-twitter-data
This dataset has over 80,000 Tweets and 12 columns. The column names are as follows:

•	user_name

•	user_location

•	user_description

•	user_created

•	user_followers

•	user_friends

•	user_fovourites

•	user_verified

•	date

•	text

•	source

•	is_retweet

The dataset had null values for the columns user_location and user_description. Hence, these columns were deleted as they do not have any influence on the sentiment analysis task. The other columns like user_created, user_followers, user_friends, user_favourites, user_verified, date, source and is_retweet is also removed from the dataset as they do not have any influence on the sentiment analysis. The “text” column is retained in the dataset as it contains the opinions of the users of Twitter about the squid game. 

**Exploratory Data Analysis:**

There are 80000 tweets in this dataset, and 12 attributes for each tweet. The dataset had null values in some of the columns. Only one column i.e., “text” is present in the dataset, as these are social media opinions, so this column needs to be prepared before any analysis. 

**Data preparation:**

A variety of pre-processing and modeling techniques were explored in order to effectively transform the text data and build a strong classifier. Data cleaning starts with checking for duplicates. All the duplicate tweets were removed from the dataset. Tweet text present in each row of the “text” column was processed by performing the following functions:
1.	Null values were removed
2.	stop words were removed
3.	punctuations were removed
4.	Weblink references were removed. 
5.	All the characters were converter to lower case.
6.	Stemming is performed as part of text normalization.

We have used VADER lexicon for sentiment analysis of text. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It is used for sentiment analysis of text which has both the polarities i.e., positive/negative. VADER not only gives the Positivity and Negativity score but also tells us about how positive or negative a sentiment is. 

**Data Visualization:**

Data visualization helps us to explore the dataset by translating the information into graphical representation. A word cloud is a data visualization tool that displays the most used words in a larger size. With the help of word cloud, most used words in squid game opinions are displayed.
Through Countplot function we were able to represent the number of occurrences of each emotion(positive, negative, and neutral) in the form of bar graph. Most common negative tweet words, positive tweet words and neutral tweet words are displayed in the form of bar graph. With the help of Word cloud, top 25 most used negative tweet words, positive tweet words and neutral tweet words are displayed. Below are some of the Data visualizations like Wordclouds, CountPlots, Bar graphs which are developed for the sentiment analysis.

![image](https://user-images.githubusercontent.com/95503789/160300022-3810c83e-9425-46b8-9f43-02284e125161.png)
![image](https://user-images.githubusercontent.com/95503789/160300030-52271b5b-52b6-4f49-9dfd-f4b421171ad2.png)
![image](https://user-images.githubusercontent.com/95503789/160300039-1176ae44-b17f-41ac-a80e-373f3e1fb563.png)
![image](https://user-images.githubusercontent.com/95503789/160300043-46a8ea77-9608-4f73-ad74-e034dcc7113a.png)
![image](https://user-images.githubusercontent.com/95503789/160300047-910d079b-64cc-42bf-8ef7-f41349e7853b.png)
![image](https://user-images.githubusercontent.com/95503789/160300051-74136413-c9ee-49bb-b059-b47c5e22ab63.png)
![image](https://user-images.githubusercontent.com/95503789/160300063-e5ed635b-92a9-454f-9666-a85eda6cadfa.png)

**Model Building:**

Packages Used:

❖	Keras

❖	Tensorflow

❖	Numpy

❖	Panda 

❖	Matplotlib

❖	SkLearn

❖	Seaborn

❖	Nltk

❖	WordCloud

❖	String

❖	Re

❖	Flask

❖	Pickle

❖	Random

**Model**

The model was built based on Keras Sequential principles. We have made used of different types of  Recurrent neural networks like SimpleRNN, LSTM, GRU and Bidirectional LSTM for building our model. Models will be evaluated on Accuracy and use the selected loss function and optimizer, since it is a multiclass classification problem. Below code chunk shows the usage of VADER lexicon for the sentiment analysis. VADER not only tells us about the positivity or negativity of the tweet but also tell us about how positive or negative a sentiment is. The VADER sentiment lexicon is sensitive both the polarity and the intensity of sentiments expressed in social media contexts. 
In order to prepare our dataset for training and validation, we need to separate it into training and validation subsets. 80/20 split was used. 

The Compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate. Threshold values are set for classifying sentences as either positive, neutral, or negative. Threshold values are:
positive sentiment : (compound score >= 0.05) 
neutral sentiment : (compound score > -0.05) and (compound score < 0.05) 
negative sentiment : (compound score <= -0.05)

**Model Comparison:**
	 
By comparing the models based on the metrics like Accuracy and loss between Training set and the test set, it is evident that performance of the GRU model is best compared to all other models. 

Below are the accuracy, loss and confusion matrix for the GRU model.

Training accuracy – 96.62% 
Validation accuracy – 94.26% 
Training loss – 0.11 
Validation loss – 0.21

![image](https://user-images.githubusercontent.com/95503789/160300264-52fe5049-8be4-4c4d-8aef-cd620da8ee39.png)
![image](https://user-images.githubusercontent.com/95503789/160300265-ec78545e-8b53-4de8-b1ed-62f1760f4ddf.png)
![image](https://user-images.githubusercontent.com/95503789/160300269-0ad0ffcc-f568-40a2-b947-ff385682b629.png)


**Model Prediction:**

The next step is to use the trained GRU model in real time to run predictions on new data. In order to do this, we had to transform the input data to tokenized sequences, similar to the way we treated our training data. 
 
**Model Deployment:**

With the help of Flask (A web framework in python that provides the tools, libraries, and technologies  necessary to build a web application), we tried to deploy our best model. Since the flask server creates a server that runs locally on the allocated runtime on google colab as localhost, we had to expose the server to the outside traffic or to make the server accessible outside the runtime globally on HTTP. For this purpose, ngrok is used. After setting up the authToken for ngrok, the model.h5 file and corresponding tokenizer.pickle file were loaded.
The function contains the model prediction code, and index.html is the landing page for our application.

**Future Scope:**

Currently our application can handle only texts in English. As the social media platform is used by people from diverse geographic regions, it is obvious that there would be significant number of tweets in other foreign languages. If those tweets can be translated and used in sentiment analysis, then our model would be able to provide accurate results and better understanding of viewer’s preference. Our sentiment analysis could be further enhanced to identify the emotion associated with each tweet. This could include emotions such as ‘Happiness’, ‘Sadness’, ‘Anger’ and ‘Disgust’. Also, it would be great if our model can be extended to utilize live tweets from twitter and find sentiments for greater business value.

**Conclusion:**

Analyzing tweets for sentiment remains an essential marketing exercise for any company looking to listen and learn from the experiences of its customers. We can use the models discussed above to get a quick overview of user sentiment. SimpleRNNs are good for processing sequence data for prediction but suffers from short-term memory. LSTM and GRUs uses gates to mitigate short-term memory issues. And they usually perform better than SimpleRNNs. Hence, we have used GRU model for sentiment prediction. Sentiment Analysis can help Netflix to be updated with their brand perception, grow their influence in the market and improve their customer service by providing quality content. Especially due to COVID, we feel that this analysis would be relevant and beneficial as the viewership has increased for Netflix.

**References:**
	
• https://www.sas.com/content/dam/SAS/support/en/sas-global-forum-proceedings/2018/2708-2018.pdf
• https://www.conviva.com/netflix-and-squid-game-using-viewership-data-to-win-the-war-on-engagement/
• https://scrapingrobot.com/blog/twitter-sentiment-analysis/
• https://github.com/Prabagini/Twitter-Sentiment-Analysis/blob/main/Jupyter%20Notebook/Twitter-Sentiment-Analysis.ipynb
• https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
• https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91
• https://colab.research.google.com/drive/1CgnLgSePdMFOPOXiN1IBXyV4oBJffxwE
• https://thecleverprogrammer.com/2021/11/03/squid-game-sentiment-analysis-using-python/

