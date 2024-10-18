# 1-3-2. 프롬프트 템플릿 (PromptTemplate)

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    # 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
    template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

    # PromptTemplate 인스턴스를 생성
    prompt_template = PromptTemplate.from_template(template_text)

    # 템플릿에 값을 채워서 프롬프트를 완성
    filled_prompt = prompt_template.format(name="홍길동", age=30)
    print(f"filled_prompt: {filled_prompt}")

    # 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)
    combined_prompt = (
            prompt_template
            + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")
            + "\n\n{language}로 번역해주세요."
    )
    filled_combined_prompt = combined_prompt.format(name="홍길동", age=30, language="영어")
    print(f"filled_combined_prompt: {filled_combined_prompt}")

    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = combined_prompt | llm | StrOutputParser()
    answer = chain.invoke({"age":30, "language":"영어", "name":"홍길동"})
    print(answer)