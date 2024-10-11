# 1-2-1. 기본 LLM 체인 (Prompt + LLM)

from dotenv import load_dotenv
import os

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


if __name__ == '__main__':
    # prompt + model + output parser
    prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
    llm = ChatOpenAI(model="gpt-4o-mini")
    output_parser = StrOutputParser()

    # LCEL chaining
    chain = prompt | llm | output_parser

    # chain 호출
    answer = chain.invoke({"input": "지구의 자전 주기는?"})
    print(f"answer: {answer}")
