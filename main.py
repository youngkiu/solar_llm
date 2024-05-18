import gradio as gr

from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma

load_dotenv()


def main():
    embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")
    embeddings = embeddings_model.embed_documents(
        [
            "What is the best season to visit Korea?",
        ]
    )

    len(embeddings), len(embeddings[0])

    sample_text_list = [
        "Korea is a beautiful country to visit in the spring.",
        "The best time to visit Korea is in the fall.",
        "Best way to find bug is using unit test.",
        "Python is a great programming language for beginners.",
        "Sung Kim is a great teacher.",
    ]

    sample_docs = [Document(page_content=text) for text in sample_text_list]

    vectorstore = Chroma.from_documents(
        documents=sample_docs,
        embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
    )

    retriever = vectorstore.as_retriever()

    result_docs = retriever.invoke("How to find problems in code?")
    print(result_docs[0].page_content[:100])

    result_docs = retriever.invoke("When to visit Korea?")
    print(result_docs[0].page_content[:100])

    result_docs = retriever.invoke("Who is a great prof?")
    print(result_docs[0].page_content[:100])

    layzer = UpstageLayoutAnalysisLoader("pdfs/kim-tse-2008.pdf", output_type="html")
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

    # 4. retrive
    retriever = vectorstore.as_retriever()
    result_docs = retriever.invoke("What is Bug Classification?")
    print(len(result_docs))
    print(result_docs[0].page_content[:100])

    llm = ChatUpstage()

    prompt_template = PromptTemplate.from_template(
        """
        Please provide most correct answer from the following context. 
        If the answer is not present in the context, please write "The information is not present in the context."
        ---
        Question: {question}
        ---
        Context: {Context}
        """
    )
    chain = prompt_template | llm | StrOutputParser()

    chain.invoke({"question": "What is bug classficiation?", "Context": result_docs})


if __name__ == "__main__":
    main()
