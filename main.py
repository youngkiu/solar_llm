import os

import gradio as gr

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from tavily import TavilyClient
import requests

load_dotenv()

layzer = UpstageLayoutAnalysisLoader("pdfs/TnC_trip_abroad_202404.pdf", output_type="html")
# For improved memory efficiency, consider using the lazy_load method to load documents page by page.
docs = layzer.load()  # or layzer.lazy_load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=1000, chunk_overlap=100, language=Language.HTML
)
splits = text_splitter.split_documents(docs)
print("Splits:", len(splits))

# 3. Embed & indexing
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
)

llm = ChatUpstage()

prompt_template = PromptTemplate.from_template(
    """
    You are an insurance assistant consultant, Solar EZ.
         Let's say there is a traveler A who is coming to Hawaii for the first time. One of family members was injured on the trip.
        Just as A is about to pay his hospital bills at a hospital in the United States, he remembers that he had Shinhan Overseas Travel Insurance Policy,
        and he hurriedly asks 'Solar EZ' about the amount of insurance compensation for injuries abroad and how to get the insurance claims.
        As a rational insurance assistant consultant, you must try your best to answer questions based on the given data, whether it is pdf or text or any other kinds of material.
        If and only if such trials fail, you are then allowed to search things up on available search engines.
        In addition, you should struggle to catch the meaning behind the consumer's quotes, although they might be hard to apprehend directly.
        Lastly, you must provide anything that will help consumers at all cost.
        you should reply things in Korean unless asked to do otherwise.
        When you get the consumer's question, you should reply to the question without any prior sentences.

        Example sentences are as the following.

주요 질문1 : 해외여행 중에 가족이 식중독 걸린거 같아요.
답변1 : 해외여행보험 약관을 볼수있는 사이트를 알려드리겠습니다.https://www.shinhanez.co.kr/static/pub/PUB2000T021.html
주요 질문2 : 약관이 뭔지 모르겠어.
답변2 : 약관은 보험계약에 관하여 보험계약자와 보험회사 상호간에 이행하여야 할 권리와 의무를 규정
한 것입니다. 상황에 맞는 보험금액을 확인하실 수 있습니다.

주요 질문3 : 약관을 볼시간이 없어. 식중독 걸렸는데 도와줘.
 답변3(PDF를 통함) : 해당 금융상품에서 해외여행중 식중독 보상금 특별약관이 있습니다.
                            해외여행 도중에 음식물의 섭취로 인해 식중독이 발생하고 그 직접적인 결과로 병원 또는 의원에 2일 이상 계속 입원하여 치료를 받은 경우(입원하지 않고 외래진료만 받은 경우는 제외) 보험금을 지급받으실 수 있습니다. 보상받으실수 있는 금액을 알려드릴까요?
주요 질문4 : 응 알려줘.
답변4 : 해외여행중 식중독 보상금은 10만원으로 보상받으실 수 있습니다. 필요한 서류를 안내해 드릴까요?
주요 질문5 : 응 지금 병원이니까 빨리 알려줘
답변5 : 보험금 청구서와 영수증을 첨부하시면 보험금 청구가 됩니다.
         자세한 사항은 홈페이지를 참조하세요. https://www.shinhanez.co.kr/static/cus/CUS50000M01.html

    ---
    Question: {question}
    ---
    Context: {context}
    """
)
chain = prompt_template | llm | StrOutputParser()


tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


solar_summary = """
SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling

We introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters,
demonstrating superior performance in various natural language processing (NLP) tasks.
Inspired by recent efforts to efficiently up-scale LLMs,
we present a method for scaling LLMs called depth up-scaling (DUS),
which encompasses depthwise scaling and continued pretraining.
In contrast to other LLM up-scaling methods that use mixture-of-experts,
DUS does not require complex changes to train and inference efficiently.
We show experimentally that DUS is simple yet effective
in scaling up high-performance LLMs from small ones.
Building on the DUS model, we additionally present SOLAR 10.7B-Instruct,
a variant fine-tuned for instruction-following capabilities,
surpassing Mixtral-8x7B-Instruct.
SOLAR 10.7B is publicly available under the Apache 2.0 license,
promoting broad access and application in the LLM field.
"""


def solar_paper_search(query: str) -> str:
    """Query for research paper about solarllm, dus, llm and general AI.
    If the query is about DUS, Upstage, AI related topics, use this.
    """
    return solar_summary



def internet_search(query: str) -> str:
    """This is for query for internet search engine like Google.
    Query for general topics.
    """
    return tavily.search(query=query)



def get_news(topic: str) -> str:
    """Get latest news about a topic.
    If users are more like recent news, use this.
    """
    # https://newsapi.org/v2/everything?q=tesla&from=2024-04-01&sortBy=publishedAt&apiKey=API_KEY
    # change this to request news from a real API
    news_url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={os.environ['NEWS_API_KEY']}"
    respnse = requests.get(news_url)
    return respnse.json()


tools = [solar_paper_search, internet_search, get_news]


llm_with_tools = llm.bind_tools(tools)


def greet(question):
    retriever = vectorstore.as_retriever()
    a = llm_with_tools.invoke(question).tool_calls
    try:
        keyword = a[0]['args'].values()
        result_docs = retriever.invoke(keyword)
        print(len(result_docs))
        print(result_docs[0].page_content[:100])
    except:
        print(a)
        result_docs = retriever.invoke(question)

    print(question)
    return chain.invoke({"question": question, "context": result_docs})


with gr.Blocks() as demo:
    gr.Image(value="./logo.webp", width=200, height=200)
    chatbot = gr.Interface(
        fn=greet,
        inputs=["text"],
        outputs=["text"],
    )

if __name__ == "__main__":
    demo.launch(share=True)
