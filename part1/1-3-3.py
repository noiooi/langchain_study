# 1-3-3. 챗 프롬프트 템플릿 (ChatPromptTemplate)

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,  HumanMessagePromptTemplate

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    # 1-4-1-1. LLM
    # MessagePromptTemplate 활용
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )
    messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
    print(messages)

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
    print(answer)

    # 1-4-1-2. Chat Model
    chat = ChatOpenAI()

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "이 시스템은 여행 전문가입니다."),
        ("user", "{user_input}"),
    ])

    chain = chat_prompt | chat
    answer = chain.invoke({"user_input": "안녕하세요? 한국의 대표적인 관광지 3군데를 추천해주세요."})
    print(answer)
