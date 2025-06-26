print("[Main] Importing...")

import os
import pandas as pd
import cooc
import articles
import kwordextractor
from importlib import reload
import gc
import torch
import intel_npu_acceleration_library
import intel_npu_acceleration_library.backend

print("[Main] Import complete.")

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

saving_directory_for_articles = "./resources/"
saving_directory_for_keywords = "./resources/keywords/"
saving_directory_for_graphs = "./resources/graphs/"

extraction_iteration_threshold = 45

#Print invalid input
def printinvalidinput():

    print("[Main] Invalid input. Try again.")

#Check if given directory exists, create if not
def check_path_exists(directory):

    #Check if path exists, create if not
    if os.path.exists(directory):
        print("[Main]", directory, "directory exists. Continuing to save...")
    else:
        print("[Main]", directory, "directory doesn't exist. Creating one...")
        os.makedirs(directory)
        print("[Main]", directory, "directory created. Continuing to save...")

#Check if input is int, try again until int
def check_input_int(what):

    #Taking input infinite loop
    while True:
        input_value = input("[Main] "+str(what)+": ")

        #Check if input is int
        try:
            input_value_int = int(input_value)
            break
        except:
            printinvalidinput()
            continue

    return input_value_int

#Check model existence
def is_there_model(path):
    required_files = [
        "config.json",
        "pytorch_model.bin",  # or "tf_model.h5" for TensorFlow
        "tokenizer_config.json",
        "vocab.txt",  # may vary depending on tokenizer type
    ]
    
    return all(os.path.isfile(os.path.join(path, f)) for f in required_files)

#Download BERT models if not found
def downloader():

    if not is_there_model("./bert-uncased-keyword-extractor"):
        print("[Main] bert-uncased-keyword-extractor not found. Downloading...")
        kwordextractor.download_model("bert-uncased-keyword-extractor")

    if not is_there_model("./bert-base-cased-news-category"):
        print("[Main] bert-base-cased-news-category not found. Downloading...")
        kwordextractor.download_model("bert-base-cased-news-category")
    
#Whether/How to customize threshold and page size - for article_retrieval_online(df)
def custom_setting():
    
    #Customization choice infinite loop
    while True:

        #Ask whether to customize retrieval settings
        print("[Main] (Additional choice) Do you want to define threshold and page size? (1/0)")
        print("[Main] Default threshold: 100, default page size: 100")
        customize_choice = str(input("[Main] Customize retrieval settings choice: "))

        #If no, go with default values
        if customize_choice == "0":

            threshold = 100
            page_size = 100
            break

        #If yes, receive inputs for threshold and page size
        elif customize_choice == "1":

            #Take threshold input and check if int
            threshold = check_input_int("Threshold")

            #Take page size input and check if int
            page_size = check_input_int("Page size")

            #Customization complete
            print("[Main] Customization complete.")
            break

        else:
            
            printinvalidinput()
            continue

    return threshold, page_size

#Article retrieval task online - for choose_dataset(df)
def article_retrieval_online(df, saving_directory):

    print("[Main] Article dataset chosen: Download recent articles online.")

    threshold, page_size = custom_setting()

    print("[Main] Beginning article retrieval process...")

    #Retrieve articles via NewsAPI
    arts = articles.get_articles(df, threshold, page_size, saving_directory)

    return arts

#Article retrieval task offline - for choose_dataset(df)
def article_retrieval_offline():

    global saving_directory_for_articles

    print("[Main] Article dataset chosen: Load downloaded articles locally")
    print("[Main] Loading local articles...")

    #Load articles locally
    arts = pd.read_csv(saving_directory_for_articles+"articles.csv")

    print("[Main] Local articles successfully loaded.")

    return arts

#Dataset choice task (second choice)
def choose_dataset(df):

    #Dataset choice infinite loop
    while True:

        #Choose dataset
        print("[Main] (Second choice) Choose article dataset: Download recent articles online(1), Load downloaded articles locally(2)")
        dataset_choice = input("[Main] Article dataset choice: ")

        #Download new articles
        if dataset_choice == "1":

            arts = article_retrieval_online(df, saving_directory_for_articles)    
            break                

        #Load previously downloaded articles
        elif dataset_choice == "2":

            arts = article_retrieval_offline()
            break

        #Invalid input for dataset choice
        else:

            printinvalidinput()
            continue

    return arts

#Task choice task (first choice)
def choose_task(df, kw, NPU_bool):

    global saving_directory_for_keywords
    global extraction_iteration_threshold

    #Task choice infinite loop
    while True:

        #Choose task
        print("[Main] (First choice) Choose task: Run new keyword extraction(1), View previous analysis(2)")
        task_choice = str(input("[Main] Task choice: "))

        #Run new keyword extraction
        if task_choice == "1":

            print("[Main] Task chosen: Run new keyword extraction.")
        
            arts = choose_dataset(df)

            #Extract keywords with BERT uncased keyword extractor
            print("[Main] Beginning keyword extraction process...")

            for i in range(int(len(arts)/extraction_iteration_threshold)+1):

                kwordextractor.load_LLMs()

                #Set interval length
                extraction_interval_start = extraction_iteration_threshold * i
                extraction_interval_end = extraction_interval_start + extraction_iteration_threshold

                #Last interval length adjustment
                if extraction_interval_end >= len(arts):

                    extraction_interval_end = len(arts) - 1

                print("[Main]", str(i+1)+"/"+str(int(len(arts)/extraction_iteration_threshold)+1), "extraction interval in process...")

                kw = kwordextractor.keyword_extraction(arts.iloc[extraction_interval_start: extraction_interval_end], kw)

                print("[Main] Saving progress to csv...")

                #Save keywords to csv
                check_path_exists(saving_directory_for_keywords)
                kwordextractor.save_keywords(kw, saving_directory_for_keywords)
                    
                kwordextractor.unload_LLMs()

                torch.cuda.empty_cache()
                #If NPU is used, emptyy NPU cache via acceleration library
                if NPU_bool:
                    intel_npu_acceleration_library.backend.clear_cache()

                gc.collect()

                reload(kwordextractor)

            print("[Main] Keyword extraction complete.")
            print("[Main] Saving to csv...")

            #Save keywords to csv
            check_path_exists(saving_directory_for_keywords)
            kwordextractor.save_keywords(kw, saving_directory_for_keywords)
            
            print("[Main] All process complete.")
            break
        
        #Skip article loading and keyword extraction
        elif task_choice == "2":
            
            print("[Main] Task chosen: View previous analysis.")
            break

        #Invalid input for task choice
        else:

            printinvalidinput()
            continue

#Final loop to display graph
def process_graphs():

    global saving_directory_for_keywords
    global saving_directory_for_graphs

    check_path_exists(saving_directory_for_graphs)

    #If input is valid
    for column in columns_kw:

        #Draw co-occurence map of chosen category
        print("[Main] Drawing co-occurence network for", column, "category.")

        cooc.get_cooc_network(saving_directory_for_keywords+column+".csv", saving_directory_for_graphs)

#Main
def main():

    global cat_index
    global cat_index_inverse

    #Print initiation status
    print("[Main] Initiating...")

    #Initiate DataFrame for articles retrival
    df = pd.DataFrame([])

    #Initiate DataFrame to feed into keyword extractor
    columns_kw = list(cat_index.keys())
    kw = pd.DataFrame(columns=columns_kw)

    #Retrieve GPU & NPU availability from kwordextractor
    GPU_bool, NPU_bool = kwordextractor.check_HW_availability()
    if GPU_bool:
        print("[Main] GPU detected via CUDA, utilizing GPU acceleration.")
    elif NPU_bool:
        print("[Main] Intel NPU detected, utilizing Intel NPU acceleration library.")
    else:
        print("[Main] GPU or NPU unavailbale.")

    print("[Main] Initiation complete.")

    while True:

        choose_task(df, kw, NPU_bool)

        process_graphs()        

if __name__ == "__main__":
    main()

#Todo list
#kwordextractor - keyword_extraction 디버깅해야함
#Co-oc 네트워크 가독성 향상