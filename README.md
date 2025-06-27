# trendtracer
**trendtracer** downloads popular and recent news articles from selected sources and performs keyword analysis, particularly in the form of [co-occurrence networks](https://en.wikipedia.org/wiki/Co-occurrence_network). 

## 📊 Results Demo
- 👉 [Click here](https://donngas.github.io/trendtracer/index.html) to see the results!

## Requirements
- Libraries listed in requirements.txt
    - Isolated installation of Torch in virtual environment may take up several gigabytes.
    - Ensure that the Torch installation is compatible with CUDA version, if applicable.
- A [NewsAPI](https://newsapi.org/) Key in .env
- [bert-uncased-keyword-extractor](https://huggingface.co/yanekyuk/bert-uncased-keyword-extractor) and [bert-base-cased-news-category](https://huggingface.co/elozano/bert-base-cased-news-category/tree/main) are automatically downloaded to directory if not found.

## Features
- Downloads top articles from selected sources via NewsAPI within free tier quota.
- Classifies news category and extracts keywords for each article with BERT models.
- Presents results as a co-occurence network for each category (top 300 keywords for visibility).
- Relative weight (importance) for each keyword & each connection (edge) is calculated and visually reflected.

## Side notes
- Utilizes NVIDIA GPU (with CUDA installed) or Intel NPU if the device supports it.
- Resets keyword extraction models between article intervals to avoid memory problems.

## Future improvements
- Seek alternative news sources (different API or crawling methods) for larger source data.
- Improve keyword extraction methods to handle words with spaces.
- Train models specifically for the purposes of this project.
- Improve code structure and library usage.
- Give articles as datasets to pipeline rather than using sequential method.

## Logs
- Initial version made in July 2024. Some parts may be deprecated or out of date.
- Code restored to working state in June 2025. Substantial refactoring still needed.
