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

load_dotenv()


def main():
    loader = PyPDFLoader("pdfs/solar_paper.pdf")
    docs = loader.load()  # or layzer.lazy_load()
    print(docs[0].page_content[:1000])

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

    chain.invoke({"question": "Explain Table 2?", "Context": docs})


if __name__ == "__main__":
    main()
