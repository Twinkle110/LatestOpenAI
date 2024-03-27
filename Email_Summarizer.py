import streamlit as st
from io import StringIO
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def generate_sentiment_analysis(text_data, system_message_content, vector_store):
    prompt = f"{system_message_content}\nPlease summarize email: {text_data}"
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4-1106-preview"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return qa({"query": prompt})

def app():
    # System message
    st.write("Please input email to summarize")

    # Text area for input
    source_text = st.text_area("Input Text", height=200)

    # Submit button
    if st.button("Submit"):
        if not source_text.strip():
            st.error(f"Please provide email to analyze.")
        else:
            # Convert text to string IO
            stringio = StringIO(source_text)
            string_data = stringio.read()

            # Text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
            )

            texts = text_splitter.split_text(string_data)

            # OpenAI embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

            # Vector store
            vector_store = FAISS.from_texts(texts, embeddings)

            # Access system message content
            system_message_content = "You are a highly skilled AI trained in email comprehension and summarization. I would like you to read the given email and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire email. Please avoid unnecessary details or tangential points.Please follow the specified sequence each in new line with bullet points : 1) Recipients, 2) Sender, 3) Subject, and 4) Email Body 5) Next Step."


            # Generate sentiment analysis
            result = generate_sentiment_analysis(string_data, system_message_content, vector_store)
            st.write("**Output:**")
            st.write(result["result"])

# Call the app function to execute it
if __name__ == '__app__':
    app()
