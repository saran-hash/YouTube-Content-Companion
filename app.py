import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,                                                                                                        CouldNotRetrieveTranscript
)
import os
from dotenv import load_dotenv
load_dotenv()

# Load HuggingFace API token from environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_youtube_transcript(url):
    try:
        # Extract video ID from URL
        yt = YouTube(url)
        video_id = yt.video_id
        # Get transcript using the correct API
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        # Combine all text from the transcript
        text = " ".join([snippet.text for snippet in transcript])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""

def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    st.set_page_config(
        page_title="Ask Questions from Video",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("Ask Questions from Video")
    st.write("Ask questions from YouTube lecture transcripts.")
    
    video_url = st.text_input("Enter YouTube Video URL", key="video_url")

    if st.button("Process Video"):
        if video_url:
            transcript_text = get_youtube_transcript(video_url)
            if transcript_text:
                save_transcript_to_file(transcript_text)

                loader = TextLoader("transcript.txt", encoding="utf-8")
                documents = loader.load()

                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever()
                # Use local model pipeline
                pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
                llm = HuggingFacePipeline(pipeline=pipe)
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

                st.session_state.qa_chain = qa_chain
                st.success("Transcript processed successfully! You can now ask questions.")
        else:
            st.warning("Please enter a valid YouTube URL.")

    if "qa_chain" in st.session_state:
        user_question = st.text_input("Ask a question from the video transcript")
        if user_question:
            result = st.session_state.qa_chain.invoke({"query": user_question})
            answer = result.get("result", str(result))
            st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()