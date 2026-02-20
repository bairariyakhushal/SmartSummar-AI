import streamlit as st
import validators
import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
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
            with st.spinner("Loading...") :
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                             headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
        
                docs=loader.load()
        
                chain=load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )
            
                summary=chain.run({"input_documents":docs})
                st.success(summary)
            
        except Exception as e:
            st.exception(f"Exception:{e}")