import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from appconfig import AppConfig
from azureai import AzureAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredFileLoader  # Load documents (e.g., PDF, DOCX, etc.)
from langchain.document_loaders import UnstructuredURLLoader
import json
import os
import sys

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

# Function to preprocess the document for extracting information
def preprocess_document(document_text):
    return document_text.strip()

# Define an efficient LLM prompt for extracting job-related details from the document
def extract_job_details_from_document(document_text):
    prompt = PromptTemplate(
        input_variables=["document_text"],
        template="""
        You are an intelligent system that extracts specific details from job-related documents.
        Extract the following fields from the provided document and return them in JSON format:
        1. Title
        2. Description
        3. Skills
        4. Location
        5. Experience
        6. Salary

        Document Text:
        {document_text}
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(document_text)

    try:
        # Safely parse the JSON response
        job_details = json.loads(response)
        return job_details
    except json.JSONDecodeError:
        print("Error: LLM response is not valid JSON. Please check the response format.")
        sys.exit("Exiting program due to invalid LLM response.")

# Define an efficient LLM prompt for extracting job-related details from the profile
def extract_profile_details(profile_text):
    prompt = PromptTemplate(
        input_variables=["profile_text"],
        template="""
        The scraped text is from a career page of a website.
        Your job is to extract the job posting details and return them in JSON format containing the
        following keys:
        Title, Description, Skills, Location, Experience, Salary.
        Profile Text:
        {profile_text}
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(profile_text)

    try:
        # Safely parse the JSON response
        profile_details = json.loads(response)
        return profile_details
    except json.JSONDecodeError:
        print("Error: LLM response is not valid JSON. Please check the response format.")
        sys.exit("Exiting program due to invalid LLM response.")

# Function to match document data with profile data and calculate matching percentage
def match_profiles_with_jd(jd_details, profile_details):
    # Combine the job details and profile details into one string for embedding comparison
    jd_combined = f"{jd_details['Title']} {jd_details['Description']} {jd_details['Skills']} {jd_details['Location']} {jd_details['Experience']} {jd_details['Salary']}"
    profile_combined = f"{profile_details['Title']} {profile_details['Description']} {profile_details['Skills']} {profile_details['Location']} {profile_details['Experience']} {profile_details['Salary']}"

    # Convert both JD and profile details into embeddings
    jd_embedding = embeddings.embed_text(jd_combined)
    profile_embedding = embeddings.embed_text(profile_combined)

    # Use Chroma DB to calculate similarity score based on embeddings
    similarity_score = chroma_db.similarity_score(jd_embedding, profile_embedding)
    match_percent = similarity_score * 100  # Normalize score to percentage

    return match_percent

def main():
    print("Welcome to Job Profile Matcher")

    # Step 1: Upload Document
    document_file = st.file_uploader("Upload a job document (PDF)", type=["pdf"])

    if document_file is None:
        print("No file uploaded. Exiting.")
        return

    # Load the document (using UnstructuredFileLoader for supported formats)
    loader = UnstructuredFileLoader(file=document_file)
    document_text = loader.load()
    document_text = preprocess_document(document_text)

    # Step 2: Extract Job Details from Document using LLM
    job_details = extract_job_details_from_document(document_text)
    if job_details is None:
        print("Failed to extract job details. Exiting.")
        return

    print("Extracted Job Details:")
    print(job_details)

    # Combine extracted job details for vectorization
    combined_job_text = preprocess_document(json.dumps(job_details))

    # Step 3: Store Job Details in Chroma DB
    chroma_db.add_texts([combined_job_text], metadatas=job_details)
    chroma_db.persist()
    print("Job Details stored in Chroma DB!")

    # Step 4: Input profile links
    print("Enter the profile links (up to 4). Leave blank to stop.")
    profile_links = []
    for i in range(4):
        link = input(f"Profile Link {i + 1}: ").strip()
        if link:
            profile_links.append(link)
        else:
            break

    if not profile_links:
        print("No profiles entered. Exiting.")
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
            print(f"Profile matched with JD. Matching Percentage: {match_percent:.2f}%")

        except Exception as e:
            print(f"Error processing URL {link}: {e}")
            sys.exit("Exiting program due to error while processing profiles.")

if __name__ == "__main__":
    main()
