# 1-2-1. 멀티 체인 (Multi-Chain)

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
    prompt2 = ChatPromptTemplate.from_template(
        "explain {english_word} using oxford dictionary to me in Korean."
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    output_parser = StrOutputParser()

    # LCEL chaining
    chain1 = prompt1 | llm | output_parser
    chain2 = (
            {"english_word": chain1}
            | prompt2
            | llm
            | StrOutputParser()
    )


    # chain 호출
    answer = chain1.invoke({"korean_word":"미래"})
    print(answer)

    answer = chain2.invoke({"korean_word":"미래"})
    print(answer)
