# trendtracer
**trendtracer** downloads popular and recent news articles from selected sources and performs keyword analysis, particularly in the form of [co-occurrence networks](https://en.wikipedia.org/wiki/Co-occurrence_network). 

## Requirements
- Libraries listed in requirements.txt
    - Isolated installation of Torch in virtual environment may take up several gigabytes.
    - Ensure that Torch installation is compatible with CUDA version, if applicable.
- [bert-uncased-keyword-extractor](https://huggingface.co/yanekyuk/bert-uncased-keyword-extractor) and [bert-base-cased-news-category](https://huggingface.co/elozano/bert-base-cased-news-category/tree/main) are automatically downloaded to directory if not found.

## Features
- Downloads top articles from selected sources via NewsAPI within free tier quota.
- Classifies news category and extracts keywords for each article with BERT models.
- Presents results as a co-occurence network for each category (top 300 keywords for visibility).
- Relative weight(importance) for each keyword & each connection(edge) is calculated and visually reflected.

## Side notes
- Utilizes NVIDIA GPU (with CUDA installed) or Intel NPU if the device supports it.
- Resets keyword extraction models between set article intervals to avoid memory problems.

## Future improvements
- Seek alternative news sources (different API or crawling methods) for larger source data.
- Train models specifically for the purposes of this project.
- Improve code structure and library usage.
- Disable/adjust model reset at article intervals when GPU is in use.
- Give articles as dataset to pipeline rather than using sequential method.

## Logs
- Initial version made in July 2024. Some parts may be deprecated or out of date.
- Code restored to working state in June 2025. Substantial refactoring still needed.
