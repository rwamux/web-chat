import streamlit as st
from langchain.agents import create_agent 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4 as bs
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import tool

st.set_page_config(page_title="Agent Demo", page_icon="🤖")
st.title("🤖 Agent Demo with LangChain and Streamlit")

llm = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"]
)

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore(embeddings)
loader = WebBaseLoader(["https://www.sicpa.com/","https://www.sicpa.com/history","https://www.sicpa.com/expertise","https://www.sicpa.com/sicpa-glance"]) 
with st.spinner("Loading documents..."):
    docs = loader.load()
    st.success(f"Loaded {len(docs)} documents.")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
document_ids = vectorstore.add_documents(texts)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer the user's query."""
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    serialized = "\n\n".join((f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs)
    
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = (
    "You are helpful assistant who has access to a tool that retrieves context from a set of documents.\n"
    "Use the tool to help answer the user queries"
)
agent = create_agent(llm, tools, system_prompt=prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the Sicpa website"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        messages = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        response = messages['messages'][-1]
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

