from transformers import pipeline
import pandas as pd
import tqdm
from transformers import AutoModel
import intel_npu_acceleration_library

#Define LLMs locally
model_1 = AutoModel.from_pretrained("./bert-uncased-keyword-extractor")
model_2 = AutoModel.from_pretrained("./bert-base-cased-news-category")

#Compile models for intel NPU
model_1 = intel_npu_acceleration_library.compile(model_1)
model_2 = intel_npu_acceleration_library.compile(model_2)

#Assign model pipelines
extractor = pipeline(task="token-classification", model="./bert-uncased-keyword-extractor")
classifier = pipeline(task="text-classification", model="./bert-base-cased-news-category")

#News category index
"""
"0": "Automobile",
"1": "Entertainment",
"2": "Politics",
"3": "Science",
"4": "Sports",
"5": "Technology",
"6": "World"
"""
cat_index = {"automobile": 0, "entertainment": 1, "politics": 2, "science": 3, "sports": 4, "technology": 5, "world": 6}
cat_index_inverse = {value: key for key, value in cat_index.items()}

#Keyword extraction function for individual article
def indv_extraction(article):

    keywords_output = extractor(article)
    category_output = classifier(article)

    #Format only keywords into a list
    keywords = [item['word'] for item in keywords_output]

    #Remove subwords that start with ##
    keywords = [word for word in keywords if not word.startswith('##')]

    #Format category into single word
    category = category_output[0]['label']

    return keywords, category

#Keyword extraction function for whole DataFrame
def keyword_extraction(tempdf, tempkw):

    global cat_index

    tempdf = tempdf[['title', 'description', 'content']]

    for i in tqdm.tqdm(range(len(tempdf))):

        key_input = tempdf.iloc[i]

        last_keyword_row, last_row_category = indv_extraction(key_input)

        #Concatenate to a dataframe of all keywords (per row) corresponding to category
        tempkw.iloc[cat_index.get(last_row_category)] = pd.concat([tempkw, pd.DataFrame(last_keyword_row)], ignore_index=True)

    return tempkw

#Save keywords to csv of each category
def save_keywords(tempkw):

    print("[Kword] Saving keywords to each category's csv...")

    for i in tqdm.tqdm(range(7)):

        tempkw.iloc[i].to_csv("./resources/keywords/"+str(cat_index_inverse.get(i))+".csv")

    print("[Kword] Successfully saved keywords to csv.")
