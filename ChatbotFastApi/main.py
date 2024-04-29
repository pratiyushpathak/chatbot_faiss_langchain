import streamlit as st
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import vectordb, chatbot
import os

def save_uploaded_file(uploaded_files, target_folder):
    """
    Save an uploaded file to a specified target folder.
 
    Parameters:
    - uploaded_file (BytesIO): The uploaded file object.
    - target_folder (str): The path to the target folder where the file will be saved.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
 
    saved_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(target_folder, uploaded_file.name)
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        saved_file_paths.append(file_path)
    return saved_file_paths

  
def main():
    st.set_page_config("Chat PDF")
    st.header("Multi-PDF Chat using Gemini")

    if 'asked_questions' not in st.session_state:
        st.session_state.asked_questions = []
    st.subheader("Question")    
    user_question = st.text_input("Enter the question")

    if user_question:
        st.session_state.asked_questions.append(user_question)
        response = chatbot.user_input(user_question)
        st.subheader("Answer")
        st.write(response["output_text"])

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Saving the uploaded file:
        
        if pdf_docs is not None:
            save_button = st.button("Save Files")
            if save_button:
                target_folder = "uploaded_files"
                saved_file_paths = save_uploaded_file(pdf_docs, target_folder)
                st.success(f"Files saved successfully at: {', '.join(saved_file_paths)}")
            
        # ==========
        
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = vectordb.get_pdf_text(pdf_docs)
                text_chunks = vectordb.get_text_chunks(raw_text)
                vectordb.get_vector_store(text_chunks)
                st.success("Done")
        
        st.subheader("Questions Asked:")
        for idx, question in enumerate(st.session_state.asked_questions):
            st.write(f"{idx + 1}. {question}")
                
if __name__ == "__main__":
    main()



                


