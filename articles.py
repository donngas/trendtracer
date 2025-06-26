import pandas as pd
from newsapi import NewsApiClient
from os import getenv
from dotenv import load_dotenv
import time

#NewsAPI Key
load_dotenv()
NAPI_KEY = str(getenv('NewsAPI_API_KEY'))
napi = NewsApiClient(api_key=NAPI_KEY)

#Ordinal suffix function for logging purposes
def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

#Function for each retrieval of articles
#Up to threshold articles per retrieval due to API quota
def each_retrieval(src, tempdf, threshold, page_size):

    #Initial request for 1st page and totalResults
    data = napi.get_everything(sort_by="popularity", sources=src, page_size=page_size)
    #Concatenate 1st page
    tempdf = pd.concat([tempdf, pd.DataFrame(data['articles'])], ignore_index=True)
    #Get totalResults
    local_em = int(data['totalResults'])
    
    #If n of articles found is larger than threshold, limit to threshold amount
    if local_em >= threshold:
        rg = range(2, int((threshold/page_size))+1)
        local_am = threshold
    else:
        local_am = local_em
        rg = range(2, int(local_am/page_size)+1)

    #Retrieval loop
    for i in rg:

        #Delay to avoid API error & calculate progress
        time.sleep(1.5)
        
        #Acquire articles and concatenate
        data = napi.get_everything(sort_by="popularity", sources=src, page_size=page_size, page=i)
        tempdf = pd.concat([tempdf, pd.DataFrame(data['articles'])], ignore_index=True)

    return local_em, local_am, tempdf

#Get articles via NewsAPI
def get_articles(tempdf, threshold, page_size, saving_directory):

    #Initiate sources list
    source = list()

    #Define sources
    all_sources = ["abc-news", "associated-press", "bbc-news", "cnn", "fox-news", "politico",
                   "the-irish-times", "reuters", "the-washington-post", "time", "vice-news", "bloomberg", 
                   "business-insider", "fortune", "the-wall-street-journal", "medical-news-today", "national-geographic", 
                   "new-scientist", "next-big-future", "hacker-news", "the-next-web", "wired"]
    for i in all_sources:
        source.append(i)
        
    #Amount of articles found in each retrieval attempt
    em = []
    #Amount of articles succesfully loaded from each retrieval
    am = []
    #Initialize both lists
    for i in range(len(all_sources)):
        em.append(0)
        am.append(0)

    #Retrieve articles each attempt at a time
    for i in range(len(source)):

        print("[Arts] Beginning", ordinal(i+1), "retrieval from", str(source[i])+".")

        try:
            em[i], am[i], tempdf = each_retrieval(source[i], tempdf, threshold, page_size)
            print("[Arts]", ordinal(i+1), "retrieval successful.", em[i], "articles found,", am[i], "articles retrieved from", str(source[i])+".")

        except Exception as e:
            em[i], am[i], tempdf = 0, 0, tempdf

            print("[Arts]", ordinal(i+1), "retrieval failed. Skipping articles from", str(source[i])+".")
            print("[Arts] Error from", ordinal(i+1), "retrieval:", e)

    #Total amount of articles found
    tas = sum(em)
    #Total amount of articles retrieved
    tar = sum(am)

    print("[Arts] All articles succesfully loaded. Total", tas, "articles found, total", tar, "articles retrieved.")
    
    tempdf.to_csv(saving_directory+"articles.csv")

    print("[Arts] Articles saved to csv.")

    return tempdf