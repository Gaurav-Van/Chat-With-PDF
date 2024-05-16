import os 
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )

    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except:
        st.warning("Issue with reading the PDF/s. Your File might be scanned so there will be nothing in chunks for embeddings to work on")

def get_response(context, question, model):

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    prompt_template = f"""
    You are a helpful and informative bot that answers questions using text from the reference context included below. \
    Be sure to respond in a complete sentence, providing in depth, in detail information and including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    If the passage is irrelevant to the answer, you may ignore it also as a Note: Based on User query Try to Look into your chat History as well
    
    Context: {context}?\n
    Question: {question}\n
    """

    try:
        response = st.session_state.chat_session.send_message(prompt_template)
        return response.text

    except Exception as e:
        st.warning(e)

def working_process(generation_config):

    system_instruction = "You are a helpful document answering assistant. You care about user and user experience. You always make sure to fulfill user request"

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

    vectorstore = st.session_state['vectorstore']

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a PDF Assistant. Ask me anything about your PDFs or Documents")
    ]
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Enter Your Query....")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            try:
                relevant_content = vectorstore.similarity_search(user_query, k=10)
                result = get_response(relevant_content, user_query, model)
                st.markdown(result)
                st.session_state.chat_history.append(AIMessage(content=result))
            except Exception as e:
                st.warning(e)

def main():

    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")

    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8000,
}

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Submit'", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing"):
                text_chunks = get_all_pdfs_chunks(pdf_docs)
                vectorstore = get_vector_store(text_chunks)
                st.session_state.vectorstore = vectorstore

    if st.session_state.vectorstore is not None:        
        working_process(generation_config)

if __name__ == "__main__":
    main()


