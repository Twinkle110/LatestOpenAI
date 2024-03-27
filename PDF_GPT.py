import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
#from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage,
)



css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/jpeg;base64,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" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,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">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# system_message_content = "You are an intelligent AI which analyses text from documents and answers the user's questions. Please answer in as much detail as possible, so that the user does not have to revisit the document. If you don't know the answer, say that you don't know, and avoid making up things."

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):

    prompt_template = """
    You are an intelligent AI which analyses text from documents and answers the user's questions. Please answer in as much detail as possible, so that the user does not have to 
    revisit the document. If you don't know the answer, say that you don't know, and avoid making up things.Please provide output in 4 sections.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    llm = ChatOpenAI(
            #model_name="gpt-3.5-turbo",
            model_name="gpt-4-1106-preview",
            #openai_api_key=OPENAI_API_KEY,
            
    )
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    question_generator_chain = LLMChain(llm=llm,prompt=prompt)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        #question_generator=question_generator_chain
    )
    
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def app():
    load_dotenv()
    #st.set_page_config(page_title="Chat with multiple PDFs",
                       #page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your document :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    app()
