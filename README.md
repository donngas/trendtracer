# trendtracer
**trendtracer** downloads popular and recent news articles from selected sources and performs keyword analysis, particularly in the form of [co-occurrence networks](https://en.wikipedia.org/wiki/Co-occurrence_network). 

## Requirements
- Libraries listed in requirements.txt
- [bert-base-cased-news-category](https://huggingface.co/elozano/bert-base-cased-news-category/tree/main) from Hugging Face
- [bert-uncased-keyword-extractor](https://huggingface.co/yanekyuk/bert-uncased-keyword-extractor) from Hugging Face

## Features
- Downloads top articles from selected sources via NewsAPI within free tier quota.
- Classifies news category and extracts keywords for each article with BERT models.
- Presents results as a co-occurence network for each category.
- Relative weight(importance) for each keyword and each connection(edge) is calculated and visually reflected.

## Side notes
- Utilizes Intel NPU if the device supports it.
- Resets keyword extraction module and related models between set article intervals to avoid memory problems.

## Future improvements
- Seek alternative news sources (different API or crawling methods) for larger source data.
- Train models specifically for the purposes of this project.
- Improve code structure and library usage
- Update anything and everything out of date.

## Logs
- Last updated in July 2024. Some parts may be deprecated or out of date.