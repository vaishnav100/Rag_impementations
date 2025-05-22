# '''
#     First install langchain using -> pip install langchain 
#     Second install google ai using -> pip install langchain_google_genai
#     Third install langchain_community using -> pip install langchain_community
#     Fourth install faiss-cpu using -> pip install faiss-cpu
# '''

from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os
import time

# Load environment variables
load_dotenv(find_dotenv())

# Configure API keys
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize embeddings (convert text to vector)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# Load and split documents
with open('python_documents.md', 'r') as f:
    content = f.read()

documents = content.split('## Document')[1:]
text_splitter = CharacterTextSplitter(separator="\n")
texts = text_splitter.create_documents(documents)

# Create FAISS vector store
vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)

# Create prompt template
template = """Use the following context to answer the question concisely. 
If you don't know, say "I don't know."

Context: {context}

Question: {question}

"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Query function with context trimming
def query_rag(question):
    start_time = time.time()
    
    # Retrieve documents
    results = qa_chain({"query": question})

    # Extract retrieved documents
    end_time = time.time()

    # Print the context and response
    print("\n Question:", question)
    print("Answer:", results['result'])
    print(f"Query time: {end_time - start_time:.4f} seconds")

question = "What is OOP?"
query_rag(question)









from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import faiss
import numpy as np
import os
import time
from langchain_core.documents import Document

# Load environment variables
load_dotenv(find_dotenv())

# Configure API keys
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize embeddings (convert text to vector)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# Load the documents from file
with open('python_documents.md', 'r') as f:
    content = f.read()

# Split the content by document headers (## Document)
documents = []
for doc in content.split('## Document')[1:]:
    documents.append(doc.strip())

# Create text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
)

# Create Document objects
docs = [Document(page_content=text) for text in documents]

# Create vector store with FAISS
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# Create a custom retriever with manual optimization
# Get the dimension from the index
dimension = vectorstore.index.d

# Create a new flat L2 index
new_index = faiss.IndexFlatL2(dimension)

# Get vectors from current index
vectors = []
for i in range(vectorstore.index.ntotal):
    vectors.append(vectorstore.index.reconstruct(i))
vectors_array = np.array(vectors)

# Add vectors to the new index
new_index.add(vectors_array)

# Replace the index
vectorstore.index = new_index

# Create the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Create custom prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Start the answer directly and remember that not getting small answer
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Initialize Gemini model
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=1
)

# Define the RAG chain using the new LCEL API
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Function to query the RAG system with timing
def query_rag(question):
    start_time = time.time()
    
    # Perform the query with our optimized chain
    result = rag_chain.invoke(question)
    
    end_time = time.time()

    print("\nQuestion:", question)
    print("Answer:", result)
    print(f"Query time: {end_time - start_time:.4f} seconds")
    
# Run the query
question = "what is OOP?"
query_rag(question)