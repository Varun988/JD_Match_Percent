import json
import os
import sys
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from azureai import AzureAI
from appconfig import AppConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create instances of AppConfig and AzureAI
config = AppConfig()
azure_ai = AzureAI(config)

# Initialize the LLM client
llm = azure_ai.get_client()

# Initialize embeddings (Hugging Face embedding model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# File and Chroma DB setup
chroma_db_name = "jd_chroma_db"
chroma_persist_dir = "chroma_db"

# Load or initialize Chroma DB
if os.path.exists(chroma_persist_dir) and os.listdir(chroma_persist_dir):
    chroma_db = Chroma(collection_name=chroma_db_name, embedding_function=embeddings, persist_directory=chroma_persist_dir)
    print("Chroma DB loaded from existing directory.")
else:
    chroma_db = Chroma(collection_name=chroma_db_name, embedding_function=embeddings, persist_directory=chroma_persist_dir)
    print("New Chroma DB initialized.")

# Function to chunk text for better processing
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Splits text into overlapping chunks for better embedding comparison."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Function to preprocess the document for extracting information
def preprocess_document(document_text):
    return document_text.strip()

# Define LLM prompt for extracting job details (no JSON format)
def extract_job_details_from_document(document_text):
    prompt = PromptTemplate(
        input_variables=["document_text"],
        template=""" 
        You are an intelligent system that extracts key details from job descriptions.
        Read the following document and extract the relevant details:

        - **Job Title**: 
        - **Description**: 
        - **Required Skills**: 
        - **Location**: 
        - **Experience Required**: 
        - **Salary Range**: 

        Ensure the extracted information is clear and formatted properly.

        Document:
        {document_text}
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(document_text)
    return response

# Define LLM prompt for extracting profile details (no JSON format)
def extract_profile_details(profile_text):
    prompt = PromptTemplate(
        input_variables=["profile_text"],
        template=""" 
        The following text is extracted from a candidate's job profile.
        Extract the following details clearly:

        - **Candidate Name (if available)**:
        - **Job Title**:
        - **Summary/Description**:
        - **Key Skills**:
        - **Location**:
        - **Years of Experience**:
        - **Expected Salary**:

        Format the response neatly for easy readability.

        Profile Text:
        {profile_text}
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(profile_text)
    return response

# Function to match document data with profile data and calculate matching percentage
def match_profiles_with_jd(jd_text, profile_text):
    jd_chunks = chunk_text(jd_text)
    profile_chunks = chunk_text(profile_text)

    max_similarity = 0  # Store the highest match

    for jd_chunk in jd_chunks:
        jd_embedding = embeddings.embed_query(jd_chunk)
        for profile_chunk in profile_chunks:
            profile_embedding = embeddings.embed_query(profile_chunk)
            similarity = cosine_similarity([jd_embedding], [profile_embedding])[0][0]
            max_similarity = max(max_similarity, similarity)

    return max_similarity * 100  # Normalize to percentage


# Function to load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    # Extract text from all pages
    document_text = "\n".join([page.page_content for page in pages])
    return document_text

# Function to load Word Document
def load_docx(file_path):
    loader = UnstructuredWordDocumentLoader(file_path)
    pages = loader.load_and_split()
    
    # Extract text from all pages
    document_text = "\n".join([page.page_content for page in pages])
    return document_text

def main():
    st.title("Job Profile Matcher")

    # Step 1: Upload Document
    document_file = st.file_uploader("Upload Job Description Document (PDF or DOCX)", type=["pdf", "docx"])

    if document_file is not None:
        # Load the document (using PyPDFLoader for PDF or UnstructuredWordDocumentLoader for DOCX)
        if document_file.name.endswith('.pdf'):
            document_text = load_pdf(document_file)
        elif document_file.name.endswith('.docx'):
            document_text = load_docx(document_file)

        document_text = preprocess_document(document_text)

        # Step 2: Extract Job Details from Document using LLM
        job_details = extract_job_details_from_document(document_text)
        if job_details is None:
            st.error("Failed to extract job details.")
            return

        st.subheader("Extracted Job Details:")
        st.write(job_details)

        # Step 3: Store Job Details in Chroma DB (with chunking)
        for chunk in chunk_text(job_details):
            chroma_db.add_texts([chunk])  # Store chunks separately

        chroma_db.persist()
        st.success("Job Details stored in Chroma DB!")

        # Step 4: Input profile links
        profile_links = []
        for i in range(4):
            profile_link = st.text_input(f"Profile Link {i + 1} (Leave blank to stop)", key=f"profile_{i}")
            if profile_link:
                profile_links.append(profile_link)
            else:
                break

        if not profile_links:
            st.error("No profiles entered. Exiting.")
            return

        # Step 5: Extract and match profile data
        for link in profile_links:
            try:
                # Use UnstructuredURLLoader to extract data from the URL
                loader = UnstructuredURLLoader(urls=[link])
                documents = loader.load()
                profile_text = " ".join(doc.page_content for doc in documents)

                # Extract details using LLM
                profile_details = extract_profile_details(profile_text)
                if profile_details is None:
                    continue  # Skip processing if extraction failed

                # Step 6: Match the profile with the uploaded JD document
                match_percent = match_profiles_with_jd(job_details, profile_details)
                st.subheader(f"Profile matched with JD: {link}")
                st.write(f"Matching Percentage: {match_percent:.2f}%")

            except Exception as e:
                st.error(f"Error processing URL {link}: {e}")
                break

if __name__ == "__main__":
    main()
