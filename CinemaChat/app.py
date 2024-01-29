import os
import imdb
import requests
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.header("🍿🎥CinemaChat: TinyLlama's IMDB Insights")
st.write(f"Get your questions' answered by 🦙TinyLlama on the movies from the IMDB reviewers' point of view.")

with st.sidebar:

    movie = st.text_input("🎞️Movie")
    api_key = os.environ['HUGGINGFACEHUB_API_KEY']

    if st.button("Submit and Process"):

        with st.spinner('⏳Fetching Reviews from IMDB website'):
            ia = imdb.IMDb()
            url = ia.get_imdbURL(ia.search_movie(movie)[0]) + 'reviews'
            response = requests.get(url)
            reviews = list()
            for review in [line for line in (response.text).split('\n') if '<div class="text show-more__control">' in line]:
                rev = ' '.join(review.split())
                reviews.append(rev.replace('<div class="text show-more__control">',"").replace("<br/><br/>","\n").replace("</div>","").replace("&#39;","'"))

        if(len(reviews)==0):
            st.write("Please type the name of the movie as stated in IMDB 🙂")
        
        else:
            with st.spinner('⏳Creating the vector database'):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
                pages = [Document(page_content=r, metadata={"source":"local"}) for r in reviews]
                splits = text_splitter.split_documents(pages)
                embedding = HuggingFaceHubEmbeddings(huggingfacehub_api_token=api_key)
                vecterstores = FAISS.from_documents(splits,embedding)
                vecterstores.save_local("faiss_index")

question = st.text_input("Type your question here 🙂! Please wait for the reply.")
template = """<|system|>
You are a personal assistant whose job is to provide answer to the user's query based
on the context for the movie {movie}. The answer should be in your own words and be in one sentence.
If you don't know the answer please simply say I don't know.
Context:
{context}</s>
<|user|>
{question}</s>
<|assistant|>\n"""

prompt = PromptTemplate(template=template, input_variables=["context","question","movie"])
llm = HuggingFaceHub(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                     model_kwargs={"temperature": 1.0},huggingfacehub_api_token=api_key)
llm_chain = LLMChain(prompt=prompt, llm=llm)

if st.button("Send Message"):

    with st.spinner('⏳Retrieving the vector embeddings'):

        embedding = HuggingFaceHubEmbeddings(huggingfacehub_api_token=api_key)
        vecterstores = FAISS.load_local("faiss_index", embedding)
        ret = vecterstores.as_retriever()

        rel_doc = ret.get_relevant_documents(question)
        context = ""
        for doc in rel_doc:
            context += doc.page_content + "\n"
        result = llm_chain.invoke({'question':question, 'context':context, 'movie':movie})

    st.write(result['text'][result['text'].find("<|assistant|>\n") + len("<|assistant|>\n"):])
            
