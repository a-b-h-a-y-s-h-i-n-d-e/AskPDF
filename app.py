import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "sentence-transformers"])

from extract_pdf import extract_text
from PyPDF2 import PdfReader
import streamlit as st 
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# some designs
st.set_page_config(page_title="Ask PDF",
                   page_icon="./assets/favicon.jpeg", layout="centered")


st.markdown(
    """
    <h1 style="text-align: center;">
        <span style="color: white;">Chat with</span> 
        <span style="color: white; background-color:#1CAC78; padding: 0 10px; border-radius: 5px;">PDF</span>
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <span style='display: block; text-align: center; color: white; font-size: 22px; margin-top: -10px; margin-bottom: 60px;'>
        Ask Anything from Your PDFs, Powered by AI.
    </span>
    """,
    unsafe_allow_html=True
)


# this are the models
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def chunk_text(text, max_length=512):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])


def main():

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:

        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_text = extract_text("uploaded_file.pdf")
        # print(extracted_text)

        question_input = st.text_input("Ask a question...")
        if question_input:

            with st.spinner("Finding the best answer...."):

                chunks = list(chunk_text(extracted_text, max_length=500))

                chunk_embeddings = embedding_model.encode(chunks)
                question_embedding = embedding_model.encode(question_input)

                similarities = util.cos_sim(question_embedding, chunk_embeddings)
                best_chunk_idx = similarities.argmax()
                most_relevant_chunk = chunks[best_chunk_idx]

                result = qa_pipeline(question=question_input, context=most_relevant_chunk)

                st.write(result['answer'])


if __name__ == "__main__":
    main()