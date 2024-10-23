# 2-2-1. 웹 문서 (WebBaseLoader)

import os
import bs4
import requests

from langchain_community.document_loaders import WebBaseLoader

# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

class CustomWebBaseLoader(WebBaseLoader):
    def _fetch_content(self, url):
        headers = {'User-Agent': os.environ['USER_AGENT']}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.content

if __name__ == '__main__':
    # 여러 개의 url 지정 가능
    url1 = "https://blog.langchain.dev/week-of-1-22-24-langchain-release-notes/"  # 404 Not Found
    url2 = "https://blog.langchain.dev/week-of-2-5-24-langchain-release-notes/"  # 404 Not Found

    loader = CustomWebBaseLoader(
        web_paths=(url1, url2),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("article-header", "article-content")
            )
        ),
    )
    docs = loader.load()
    print(len(docs))