import streamlit as st
from io import StringIO
import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

def generate_minutes(text_data, system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease summarize the following text for a meeting minute: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please upload a .txt file to generate minutes of the meeting.")

    uploaded_file = st.file_uploader("Choose a .txt file", "txt")

    # Enable button only if file is uploaded
    if uploaded_file is not None:
        submit_button = st.button('Generate Output')
    else:
        submit_button = None

    if submit_button:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
        )

        texts = text_splitter.split_text(string_data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vector_store = FAISS.from_texts(texts, embeddings)

        # Access system message content
        system_message_content = "You are an AI assistant tasked with generating Meeting Minutes (MOM) from the provided text, including details such as Participants, Date, and Time. Additionally, you are responsible for conducting Sentiment Analysis on the text and producing a 'Who Said What' section with insightful details. Please ensure to follow the specified sequence: 1) Minutes of Meeting, 2) Sentiment Analysis, 3) Who Said What, 4) Next Steps" 

        result = generate_minutes(string_data, system_message_content, vector_store)
        st.write("**Output:**")
        st.write(result["result"])
    else:
        st.warning("Please upload a document to proceed.")

# Call the app function to execute it
if __name__ == '__app__':
    app()
