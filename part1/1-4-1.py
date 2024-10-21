# 1-4-1. LangChain 모델 유형

from dotenv import load_dotenv
import os

from langchain_openai import OpenAI

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    llm = OpenAI()

    answer = llm.invoke("한국의 대표적인 관광지 3군데를 추천해주세요.")
    print(answer)

