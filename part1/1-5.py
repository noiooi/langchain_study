# 1-5. 출력 파서 (Output Parser)

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1-5-1. CSV Parser
    # output_parser = CommaSeparatedListOutputParser()
    # format_instructions = output_parser.get_format_instructions()
    # print(format_instructions)
    #
    # prompt = PromptTemplate(
    #     template="List five {subject}.\n{format_instructions}",
    #     input_variables=["subject"],
    #     partial_variables={"format_instructions": format_instructions},
    # )
    # chain = prompt | llm | output_parser
    # answer = chain.invoke({"subject": "popular Korean cusine"})
    # print(f"answer: {answer}")

    # 1-5-2. JSON Parser
    # 자료구조 정의 (pydantic)
    class CusineRecipe(BaseModel):
        name: str = Field(description="name of a cusine")
        recipe: str = Field(description="recipe to cook the cusine")

    # 출력 파서 정의
    output_parser = JsonOutputParser(pydantic_object=CusineRecipe)
    format_instructions = output_parser.get_format_instructions()
    # print(format_instructions)

    # prompt 구성
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )
    chain = prompt | llm | output_parser
    answer = chain.invoke({"query": "Let me know how to cook Bibimbap"})
    print(f"answer: {answer}")
