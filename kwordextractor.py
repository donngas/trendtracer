from transformers import pipeline, BertForTokenClassification, BertForSequenceClassification, BertTokenizer
import pandas as pd
import tqdm
import intel_npu_acceleration_library
import torch
import gc
import time
import intel_npu_acceleration_library.backend

GPU_bool = False
NPU_bool = False

'''
#Check GPU & NPU availability
if torch.cuda.is_available():
    GPU_bool = True
elif intel_npu_acceleration_library.backend.npu_available():
    NPU_bool = True
'''

#Check NPU availability
def check_HW_availability():
    global GPU_bool, NPU_bool
    #Check GPU & NPU availability
    if torch.cuda.is_available():
        GPU_bool = True
    elif intel_npu_acceleration_library.backend.npu_available():
        NPU_bool = True
    return GPU_bool, NPU_bool

#Download model when called by main
def download_model(model_id):
    if model_id == "bert-uncased-keyword-extractor":
        model = BertForTokenClassification.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
        model.save_pretrained("./bert-uncased-keyword-extractor")
        tokenizer = BertTokenizer.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
        tokenizer.save_pretrained("./bert-uncased-keyword-extractor")
        
    elif model_id == "bert-base-cased-news-category":
        model = BertForSequenceClassification.from_pretrained("elozano/bert-base-cased-news-category")
        model.save_pretrained("./bert-base-cased-news-category")
        tokenizer = BertTokenizer.from_pretrained("elozano/bert-base-cased-news-category")
        tokenizer.save_pretrained("./bert-base-cased-news-category")

#Load LLMs
def load_LLMs():

    global model_1
    global model_2
    global tokenizer_1
    global tokenizer_2
    global extractor
    global classifier

    global GPU_bool
    global NPU_bool

    #Define LLMs locally
    model_1 = BertForTokenClassification.from_pretrained("./bert-uncased-keyword-extractor")
    tokenizer_1 = BertTokenizer.from_pretrained("./bert-uncased-keyword-extractor")
    model_2 = BertForSequenceClassification.from_pretrained("./bert-base-cased-news-category")
    tokenizer_2 = BertTokenizer.from_pretrained("./bert-base-cased-news-category")

    if GPU_bool:
        device = torch.device("cuda")
        model_1.to(device)
        model_2.to(device)
    elif NPU_bool:
        #Compile models for intel NPU
        model_1 = intel_npu_acceleration_library.compile(model_1, dtype=torch.int8)
        model_2 = intel_npu_acceleration_library.compile(model_2, dtype=torch.int8)
    else:
        device = torch.device("cpu")

    #Assign model pipelines
    if GPU_bool:
        extractor = pipeline(task="token-classification", model=model_1, tokenizer=tokenizer_1, device=0)
        classifier = pipeline(task="text-classification", model=model_2, tokenizer=tokenizer_2, device=0)
        print("[Kword] Model set to GPU.")
    elif NPU_bool:
        extractor = pipeline(task="token-classification", model=model_1, tokenizer=tokenizer_1)
        classifier = pipeline(task="text-classification", model=model_2, tokenizer=tokenizer_2)
        print("[Kword] Model set to NPU.")
    else:
        extractor = pipeline(task="token-classification", model=model_1, tokenizer=tokenizer_1, device=-1)
        classifier = pipeline(task="text-classification", model=model_2, tokenizer=tokenizer_2, device=-1)
        print("[Kword] Model set to CPU.")

#Unload LLMs
def unload_LLMs():

    global model_1
    global model_2
    global tokenizer_1
    global tokenizer_2
    global extractor
    global classifier

    #Delete models, tokenizers, and pipelines upon confirming existence
    if 'model_1' in globals():
        del model_1
    if 'model_2' in globals():
        del model_2
    if 'tokenizer_1' in globals():
        del tokenizer_1
    if 'tokenizer_2' in globals():
        del tokenizer_2
    if 'extractor' in globals():
        del extractor
    if 'classifier' in globals():
        del classifier

    torch.cuda.empty_cache()
    gc.collect()
    #If NPU is used, emptyy NPU cache via acceleration library
    if NPU_bool:
        intel_npu_acceleration_library.backend.clear_cache()

    time.sleep(1)

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
columns_kw = list(cat_index.keys())

#Ordinal suffix function for logging purposes
def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

#Find first available row for designated column
def find_first_empty_row(tempkw, column):

    #Iterate through rows to find empty one
    for i in range(len(tempkw)):
        if pd.isna(tempkw.at[i, column]):
            return i
        
    #If there's no empty one, return next one
    return len(tempkw)

#Keyword extraction function for individual article
def indv_extraction(article):

    keywords_output = extractor(article)
    category_output = classifier(article)

    #Format only keywords into a list
    keywords = [item['word'] for item in keywords_output]

    #Remove subwords that start with ##
    keywords = [word for word in keywords if not word.startswith('##')]

    #Remove words that is a single comma
    keywords = [item for item in keywords if item != ',']

    #Reformat list to ['word1, word2, word3,..., wordn']
    keywords_reform = ', '.join(keywords)

    #Format category into single word
    category = category_output[0]['label']
    category = category.lower()

    return keywords_reform, category

#Keyword extraction function for whole DataFrame
def keyword_extraction(tempdf, tempkw):

    global cat_index
    global columns_kw
    errors = 0

    tempdf = tempdf[['title', 'description', 'content']]

    #Keyword extraction iteration
    for i in tqdm.tqdm(range(len(tempdf))):

        #Try to extract keyword for individual article
        try:

            #Pull each article from articles DataFrame
            key_input = tempdf.iloc[i]
            
            #Convert planned input to appropriate format for BERT
            key_input = "{'title': '"+str(key_input['title'])+"', 'description': '"+str(key_input['description'])+"', 'content': '"+str(key_input['content'])+"'}"

            #Run individual extraction
            last_keyword_row, last_row_category = indv_extraction(key_input)

            #Convert last_keyword_row to appropriate format to merge with tempkw, assign column
            last_keyword_row = pd.DataFrame({last_row_category: last_keyword_row}, index=[0])

            #Find first empty row for that column
            row_index = find_first_empty_row(tempkw, last_row_category)

            #If row_index is beyond the current length of tempkw, append a new row
            if row_index >= len(tempkw):
                new_row = pd.Series(dtype='object')
                tempkw = pd.concat([tempkw, pd.DataFrame([new_row])], ignore_index=True)

            #Merge last row with tempkw
            tempkw.at[row_index, last_row_category] = last_keyword_row.at[0, last_row_category]

        #Handle exceptions during keyword extraction
        except Exception as error_during_extraction:
            tempkw = tempkw
            print("[Kword] DEBUG: Error:", error_during_extraction)
            if '0xe06d7363' in str(error_during_extraction):
                break
            errors += 1

    print("[Kword]", errors, "exceptions occured during keyword extraction. Accountable rows were skipped, if any.")

    return tempkw

#Save keywords to csv of each category
def save_keywords(tempkw, saving_directory):

    global columns_kw

    print("[Kword] Saving keywords to each category's csv...")

    #Save each category to csv
    for i in tqdm.tqdm(columns_kw):

        tempkw[i].to_csv(saving_directory+i+".csv")

    print("[Kword] Successfully saved keywords to csv.")
