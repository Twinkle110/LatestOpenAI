import streamlit as st

def app():
    #st.set_page_config(
     #   page_title="TA GPT",
      #  page_icon="🤖",
    #)

    st.header("People Success Chatbot")
    st.sidebar.success("Select a page above.")

    # Add "Talent Analytics Chatbot" as content
    #st.header("Talent Analytics Chatbot")
    st.write("""Welcome to the People Success Chatbot. I am an intelligent chatbot created by combining the strengths of Langchain and Streamlit. I use large language models to provide context-sensitive interactions. My goal is to help you better understand your data.
      I support Image, PDF and Text prompt transcript 🧠""")
    
     
    st.write("""Here, you can interact with our chatbot to get insights on your data""")

    st.write("**Created by Global Talent Analytics**")
    #Contact
    with st.sidebar.expander("📬 Contact"):

     st.write("**Email:**",
    "[Global Talent Analytics](twinklepardeshi8@gmail.com)")
 

    if __name__ == '__app__':
      app()
