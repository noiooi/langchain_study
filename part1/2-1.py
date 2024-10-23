# 2-1. RAG 개요

from dotenv import load_dotenv
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# Set the API key as an environment variable
api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    # Step1. Data Loader - 웹페이지 데이터 가져오기
    # 위키피디아 정책과 지침
    url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
    loader = WebBaseLoader(url)

    # 웹페이지 텍스트 -> Documents
    docs = loader.load()

    # print(len(docs)) # 문서 객체는 1개만 존재
    # print(len(docs[0].page_content))  # 문자열의 문자 개수
    # print(docs[0].page_content[5000:6000])

    # Step2. Text Split (Documents -> small chunks: Documents)
    # 문장을 최대 1000글자 단위로 분할, 200글자는 각 분할마다 겹치게 하여 문액이 유지되도록 처리
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # print(len(splits))
    # print(splits[10])

    # Step3. Indexing (Texts -> Embedding -> Store)
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=OpenAIEmbeddings())

    # docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
    # print(len(docs))
    # print(docs[0].page_content)

    # Step4. 검색(Retrieval)
    # LangChain의 retriever 메소드를 사용

    # Step5. 생성(Generation)
    # Prompt
    template = '''Answer the question based only on the following context:
    {context}
    
    Question: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever()

    # Combine Documents
    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    # RAG Chain 연결
    rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # Chain 실행
    answer = rag_chain.invoke("격하 과정에 대해서 설명해주세요.")
    print(answer)