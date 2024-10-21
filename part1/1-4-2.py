# 1-4-2. LangChain의 LLM 모델 파라미터 설정

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    # 1-4-2-1. LLM 모델에 직접 파라미터를 전달
    # 모델 파라미터 설정
    params = {
        "temperature": 0.7,         # 생성된 텍스트의 다양성 조정
        "max_tokens": 100,          # 생성할 최대 토큰 수
        "frequency_penalty": 0.5,   # 이미 등장한 단어의 재등장 확률
        "presence_penalty": 0.5,    # 새로운 단어의 도입을 장려
        "stop": ["\n"]              # 정지 시퀀스 설정

    }

    # 모델 인스턴스를 생성할 때 설정
    model = ChatOpenAI(model="gpt-4o-mini", **params)

    # 모델 호출
    question = "태양계에서 가장 큰 행성은 무엇인가요?"
    response = model.invoke(input=question)

    # 문자열 출력
    print(response.content)

    # 1-4-2-2. LLM 모델 파라미터를 추가로 바인딩 (bind 메소드)
    model = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
        ("user", "{user_input}"),
    ])
    messages = prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
    before_answer = model.invoke(messages)

    # binding 이전 출력
    print(f"binding 이전 출력: {before_answer}")

    # 모델 호출 시 추가적인 인수를 전달하기 위해 bind 메서드 사용 (응답의 최대 길이를 10 토큰으로 제한)
    chain = prompt | model.bind(max_tokens=10)
    after_answer = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

    # binding 이후 출력
    print(f"binding 이후 출력: {after_answer}")
