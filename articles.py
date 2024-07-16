import pandas as pd
from newsapi import NewsApiClient
import os
import time
import tqdm

#NewsAPI Key
NAPI_KEY = os.environ.get('NewsAPI_API_KEY')
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
def each_retrieval(src, tempdf):

    #Define threshold and page size
    threshold = 100
    ps = 100

    #Initial request for 1st page and totalResults
    data = napi.get_everything(sort_by="popularity", sources=src, page_size=ps)
    #Concatenate 1st page
    tempdf = pd.concat([tempdf, pd.DataFrame(data['articles'])], ignore_index=True)
    #Get totalResults
    local_em = int(data['totalResults'])
    
    #If n of articles found is larger than threshold, limit to threshold amount
    if local_em >= threshold:
        rg = range(2, int((threshold/ps))+1)
        local_am = threshold
    else:
        rg = range(2, int(local_am/ps)+1)
        local_am = local_em

    #Retrieval loop
    for i in tqdm.tqdm(rg):

        #Delay to avoid API error & calculate progress
        time.sleep(1.5)
        
        #Acquire articles and concatenate
        data = napi.get_everything(sort_by="popularity", sources=src, page_size=ps, page=i)
        tempdf = pd.concat([tempdf, pd.DataFrame(data['articles'])], ignore_index=True)
    
    return local_em, local_am, tempdf

#Get articles via NewsAPI
def get_articles(tempdf):

    #Initiate sources list
    source = list()

    #Define sources
    all_sources = ["abc-news", "associated-press", "bbc-news", "cnn", "fox-news", "geogle-news", "politico",
                   "reddit-r-all", "reuters", "the-washington-post", "time", "vice-news", "bloomberg", 
                   "business-insider", "fortune", "the-wall-street-journal", "medical-news-today", "national-geographic", 
                   "new-scientist", "next-big-future", "hacker-news", "the-next-web", "wired"]
    for i in all_sources:
        source.append(i)
        
    #Amount of articles found in each retrieval attempt
    em = []
    #Amount of articles succesfully loaded from each retrieval
    am = []

    #Retrieve articles each attempt at a time
    for i in range(len(source)):

        print("[Arts] Beginning", ordinal(i+1), "retrieval.")

        em[i], am[i], tempdf = each_retrieval(source[i], tempdf)

        print(ordinal(i+1), "[Arts] retrieval successful.", em[i], "articles found,", am[i], "articles retrieved.")

    #Total amount of articles found
    tas = sum(em)
    #Total amount of articles retrieved
    tar = sum(am)

    print("[Arts] All articles succesfully loaded. Total", tas, "articles found, total", tar, "articles retrieved.")
    
    tempdf.to_csv("./resources/articles.csv")

    print("[Arts] Articles saved to csv.")

    return tempdf