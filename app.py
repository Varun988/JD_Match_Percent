import streamlit as st
import pandas as pd
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from appconfig import AppConfig
from azureai import AzureAI

# Create instances of AppConfig and AzureAI
config = AppConfig()
azure_ai = AzureAI(config)

import sys  # Import sys to enable program termination
import json
import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader

# Initialize the LLM client
llm = azure_ai.get_client()

# Function to preprocess and combine JD fields
def preprocess_jd(row):
    combined_text = f"""
    Title: {row.get('Title', 'N/A')}
    Description: {row.get('Description', 'N/A')}
    Skills: {row.get('Skills', 'N/A')}
    Location: {row.get('Location', 'N/A')}
    Experience: {row.get('Experience', 'N/A')}
    Salary: {row.get('Salary', 'N/A')}
    """
    return combined_text.strip()

# Initialize embeddings
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

def extract_profile_details(profile_text):
    # Define the LLM and prompt for profile extraction
    prompt = PromptTemplate(
        input_variables=["profile_text"],
        template="""
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the
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
        # Handle invalid JSON response
        print("Error: LLM response is not valid JSON. Please check the response format.")
        print(f"LLM Response: {response}")
        sys.exit("Exiting program due to invalid LLM response.")

def main():
    print("Welcome to Job Profile Matcher")

    # Step 1: Upload CSV file
    jd_file_path = input("Enter the path to the JD CSV file: ").strip()

    if not os.path.exists(jd_file_path):
        print("File not found. Please check the path and try again.")
        return

    # Load JD file
    jd_df = pd.read_csv(jd_file_path)
    print("Job Descriptions Loaded:")
    print(jd_df.head())

    # Combine JD columns for vectorization
    jd_df["Combined"] = jd_df.apply(preprocess_jd, axis=1)

    # Step 2: Store JDs in Chroma DB
    if not os.path.exists(chroma_persist_dir) or not os.listdir(chroma_persist_dir):
        store_jds = input("Do you want to store these JDs in the vector database? (yes/no): ").strip().lower()
        if store_jds == "yes":
            for _, row in jd_df.iterrows():
                jd_text = row["Combined"]
                chroma_db.add_texts([jd_text], metadatas={"Title": row["Title"]})

            # Persist the database
            chroma_db.persist()
            print("Job Descriptions stored in Chroma DB!")
    else:
        print("Job Descriptions are already stored in Chroma DB.")

    # Step 3: Input profile links
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

    # Step 4: Extract details and store profiles in Chroma DB
    for link in profile_links:
        try:
            # Use UnstructuredURLLoader to extract data from the URL
            loader = UnstructuredURLLoader(urls=[link])
            documents = loader.load()
            profile_text = " ".join(doc.page_content for doc in documents)

            # Extract details using LLM
            profile_details = extract_profile_details(profile_text)
            if profile_details is None:
                continue  # Skip storing if extraction failed

            # Combine extracted details into a single text block
            combined_profile_text = preprocess_jd(pd.Series(profile_details))

            # Store in Chroma DB
            chroma_db.add_texts([combined_profile_text], metadatas=profile_details)
        except Exception as e:
            print(f"Error processing URL {link}: {e}")
            sys.exit("Exiting program due to error while processing profiles.")

    chroma_db.persist()
    print("Profiles stored in Chroma DB!")

    # Step 5: Match profiles with JDs
    match_profiles = input("Do you want to match profiles with the stored JDs? (yes/no): ").strip().lower()
    if match_profiles == "yes":
        matching_results = []
        for _, row in jd_df.iterrows():
            jd_text = row["Combined"]
            matches = chroma_db.similarity_search(jd_text, top_k=3)  # Adjust top_k as needed
            for match in matches:
                match_percent = match["score"] * 100  # Normalize score to percentage
                matching_results.append((jd_text, match["metadata"].get("Title", "Unknown"), match_percent))

        # Display results
        for jd_text, title, percent in matching_results:
            print(f"JD: {jd_text}")
            print(f"Matched Profile Title: {title}")
            print(f"Matching Percentage: {percent:.2f}%")

if __name__ == "__main__":
    main()
