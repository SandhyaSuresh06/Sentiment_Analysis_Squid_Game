# Sentiment_Analysis_Squid_Game

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
