import os
import streamlit as st
import pickle
import time
from langchain.llms.vertexai import VertexAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
# import faiss
from dotenv import load_dotenv

load_dotenv()

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
  url = st.sidebar.text_input(f"URL {i+1}")
  urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = 'faiss_store'

main_placeholder = st.empty()
llm = VertexAI(temperature=0.9, max_tokens=500)
model_name = "textembedding-gecko@003" 
embeddings = VertexAIEmbeddings(model_name=model_name)

if process_url_clicked:
  if not os.path.exists(file_path):
    os.makedirs(file_path)
  
  # load data
  loader = SeleniumURLLoader(urls=urls)
  main_placeholder.text("Loading data...")
  data = loader.load()

  # split data
  text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '. ', ','],
    chunk_size=1000
  )
  main_placeholder.text("Text Splitter...Started...")
  docs = text_splitter.split_documents(data)
  
  if not docs:
    st.error("No documents found. Please check the URLs provided.")
  else:
    # create embeddings
    vectorstore_vertexai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    # https://stackoverflow.com/questions/77605224/cannot-pickle-thread-rlock-object-while-serializing-faiss-object
    vectorstore_vertexai.save_local(file_path)


query = main_placeholder.text_input("Questions: ")
if query:
  if os.path.exists(file_path):
      vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
      chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
      result = chain({"question": query}, return_only_outputs=True)
      st.header("Answer:")
      st.subheader(result["answer"])
      
      sources = result.get("sources", "")
      if sources:
        st.header("Sources:")
        sources_list = sources.split("\n")
        for source in sources_list:
          st.write(source)