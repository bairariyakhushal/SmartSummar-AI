import streamlit as st
import validators
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

st.title("Summarize Text From YT Video or Any Website")

generic_url=st.text_input("URL",label_visibility="collapsed")

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=GROQ_API_KEY,model="llama-3.3-70b-versatile")

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(
    input_variables=['text'],
    template=prompt_template
)

if st.button("Summarize") :
    if not generic_url.strip() :
        st.error("Please provide the URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url")
    else:
        try:
            with st.spinner("Loading..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                        docs = loader.load()
                    except Exception:
                        # Fallback: use pytubefix captions when YouTube blocks the transcript API
                        try:
                            from pytubefix import YouTube
                            yt = YouTube(generic_url)
                            captions = yt.captions
                            cap = (captions.get("en") or captions.get("a.en")
                                   or next(iter(captions.values()), None))
                            if cap:
                                docs = [Document(page_content=cap.generate_srt_captions())]
                            else:
                                st.error("No transcript is available for this YouTube video.")
                                st.stop()
                        except Exception as yt_err:
                            st.error(f"Could not retrieve YouTube transcript: {yt_err}")
                            st.stop()
                else:
                    loader = WebBaseLoader(
                        web_paths=[generic_url],
                        requests_kwargs={
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                            }
                        }
                    )
                    docs = loader.load()

                if not docs or not any(doc.page_content.strip() for doc in docs):
                    st.error("Could not extract content from this URL. The page may require JavaScript or block automated access.")
                else:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run({"input_documents": docs})
                    st.success(summary)

        except Exception as e:
            st.exception(f"Exception: {e}")