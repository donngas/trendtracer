from newsapi import NewsApiClient
import os
import pandas as pd
import cooc
import articles
import kwordextractor

#NewsAPI Key
NAPI_KEY = os.environ.get('NewsAPI_API_KEY')
napi = NewsApiClient(api_key=NAPI_KEY)

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
   
#Main
def main():

    global cat_index
    global cat_index_inverse

    #Print initiation status
    print("[Main] Initiating...")

    #Initiate DataFrame for articles retrival
    df = pd.DataFrame([])

    #Initiate DataFrame to feed into keyword extractor
    kw = pd.DataFrame([[], [], [], [], [], [], []])

    print("[Main] Initiation complete.")

    #Main infinite loop
    while True:

        #Choose task
        print("[Main] Choose task: Run new keyword extraction(1), View previous analysis(2)")
        task_choice = input("[Main] Task choice: ")

        #Run new keyword extraction
        if task_choice == 1:

            print("[Main] Task chosen: Run new keyword extraction.")
            
            #Choose dataset
            print("[Main] Choose article dataset: Download recent articles online(1), Load downloaded articles locally(2)")
            dataset_choice = input("[Main] Article dataset choice: ")

            #Download new articles
            if dataset_choice == 1:

                print("[Main] Article dataset chosen: Download recent articles online.")
                print("[Main] Beginning article retrieval process...")

                #Retrieve articles via NewsAPI
                arts = articles.get_articles(df)

            #Load previously downloaded articles
            elif dataset_choice == 2:

                print("[Main] Article dataset chosen: Load downloaded articles locally")
                print("[Main] Loading local articles...")

                #Load articles locally
                arts = pd.read_csv("./resources/articles.csv")

                print("[Main] Local articles successfully loaded.")

            #Invalid input for dataset choice
            else:

                print("[Main] Invalid input. Try again.")
                continue

            #Extract keywords with BERT uncased keyword extractor
            print("[Main] Beginning keyword extraction process...")

            kw = kwordextractor.keyword_extraction(arts, kw)

            print("[Main] Keyword extraction complete.")
            print("[Main] Saving to csv...")

            #Save keywords to csv
            kwordextractor.save_keywords(kw)
            
            print("[Main] All process complete.")
        
        
        #Skip article loading and keyword extraction
        elif task_choice == 2:
            
            print("[Main] Task chosen: View previous analysis.")

        #Invalid input for task choice
        else:

            print("[Main] Invalid input. Try again.")
            continue

        #Loop to display results
        while True:

            #Choose article category
            print("[Main] Available categories are: Automobile(1), Entertainment(2), Politics(3), Science(4), Sports(5), Technology(6), World(7).")
            input_category = int(input("[Main] Enter category number: "))
            input_category_str = str(cat_index_inverse.get(input_category-1))

            #Draw co-occurence map of chosen category
            print("[Main] Drawing co-occurence network for", input_category_str, "category.")

            cooc.get_cooc_network("./resources/keywords/"+input_category_str+".csv")

            print("[Main] Successfully displayed co-occurence network for", input_category_str, "category.")

if __name__ == "__main__":
    main()

#Todo list
#기사 데이터베이스 분리, 가능하면 오프라인 처리 (매번 다운로드할겨?)
#Co-oc 네트워크 가독성 향상