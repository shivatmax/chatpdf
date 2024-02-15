import streamlit as st
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    return "".join([page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages])

def get_text_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50,length_function=len).split_text(text)

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=openai_api_key)
    FAISS.from_texts(text_chunks, embedding=embeddings).save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
Please provide a comprehensive and detailed answer to the question, strictly using only the information provided in the context. Avoid introducing any external information or assumptions not directly found in the given context.

Context:
{context}

Question:
{question}

Your answer should be directly informed by and confined to the details present in the context. 

Answer:
"""
    llm = ChatOpenAI(model_name="gpt-4-0125-preview", api_key=openai_api_key, temperature=0.2, max_tokens=1024)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question, search_kwargs={"k": 15}, return_texts=True )
    print(docs)
    chain = get_conversational_chain()
    with get_openai_callback() as cb:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(cb)
    st.write("Reply:\n", response["output_text"])

def setup_page():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with Multiple PDFs")

def get_user_question():
    user_question = st.text_input("Ask a Question from the PDF Files", help="Enter your question here and get answers based on the uploaded PDF content.")
    if user_question:
        st.success("You can now click the 'Submit' button to get an answer.")
    else:
        st.warning("Please enter a question.")
    return user_question

def process_files(pdf_docs):
    if pdf_docs:
        with st.spinner("Extracting text and processing PDF files..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        st.success("Done! You are now ready to ask questions.")
        st.balloons()

def display_sidebar():
    with st.sidebar:
        st.title("Menu")

        def store_api_key():
            global openai_api_key
            openai_api_key = st.text_input("Enter your OpenAI API key:")
            if not openai_api_key:
                st.warning("Please enter your OpenAI API key.")

        store_api_key()
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, help="Upload one or more PDF files from which you want to extract information.")
        if st.button("Submit & Process"):
            process_files(pdf_docs)

def main():
    setup_page()
    display_sidebar()
    
    user_question = get_user_question()
    if user_question:
        try:
            with st.spinner("Processing your question..."):
                user_input(user_question)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
