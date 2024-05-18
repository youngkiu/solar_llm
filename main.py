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

load_dotenv()

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


def greet(question):
    # 4. retrieve
    retriever = vectorstore.as_retriever()
    result_docs = retriever.invoke(question)
    print(len(result_docs))
    print(result_docs[0].page_content[:100])

    return chain.invoke({"question": question, "Context": result_docs})


with gr.Blocks() as demo:
    chatbot = gr.Interface(
        fn=greet,
        inputs=["text"],
        outputs=["text"],
    )

if __name__ == "__main__":
    demo.launch()
