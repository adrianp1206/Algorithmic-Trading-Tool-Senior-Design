from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_scores(sentence):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", round(sentiment_dict['neg']*100, 2), "% Negative")
    print("sentence was rated as ", round(sentiment_dict['neu']*100, 2), "% Neutral")
    print("sentence was rated as ", round(sentiment_dict['pos']*100, 2), "% Positive")
 
    print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.2 :
        print("Positive")
 
    elif sentiment_dict['compound'] <= - 0.2 :
        print("Negative")
 
    else :
        print("Neutral")

if __name__ == '__main__':
    News = ["WASHINGTON, Oct 18 (Reuters) - The U.S. auto safety regulator on Friday opened an investigation into 2.4 million Tesla vehicles equipped with the automaker's Full Self-Driving (FSD) software after four reported collisions, including a 2023 fatal crash."]
    sentiment_scores(News)

