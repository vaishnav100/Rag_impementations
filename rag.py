from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for RAG components
vectorstore = None
qa_chain = None
llm = None

def initialize_rag():
    """Initialize the RAG system components"""
    global vectorstore, qa_chain, llm
    
    try:
        # Configure API keys
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

        # Load and split documents
        if not os.path.exists('python_documents.md'):
            raise FileNotFoundError("python_documents.md file not found")
            
        with open('python_documents.md', 'r', encoding='utf-8') as f:
            content = f.read()

        documents = content.split('## Document')[1:]
        if not documents:
            raise ValueError("No documents found in the file")
            
        text_splitter = CharacterTextSplitter(separator="\n")
        texts = text_splitter.create_documents(documents)

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)

        # Create prompt template
        template = """
            You are an AI assistant helping answer questions based on the provided documentation. Use only the information in the context to formulate a clear and concise response.

            ---------------------
            Context:
            {context}
            ---------------------

            Question:
            {question}

            Instructions:
            - If the answer is not explicitly found in the context, say: "I'm only able to provide answers based on the provided document, and no relevant information was found."
            - Keep your answer factual and avoid speculation.
            - Prefer bullet points for clarity if multiple points are involved.
            - Do not repeat the question in the answer.
        """


        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

        # Initialize LLM
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.1
        )

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

def truncate_context(docs, max_tokens=1024):
    """Truncate context to fit within token limits"""
    context = ""
    token_count = 0
    for doc in docs:
        doc_tokens = len(doc.split())
        if token_count + doc_tokens > max_tokens:
            break
            
        context += doc + "\n"
        token_count += doc_tokens
    
    if not context.strip():
        context = "No relevant information found."
    
    return context

def query_rag_with_built_in_chain(question):
    """Query using the built-in RetrievalQA chain"""
    try:
        start_time = time.time()
        results = qa_chain({"query": question})
        end_time = time.time()
        
        return {
            "answer": results["result"],
            "query_time": round(end_time - start_time, 4),
            "method": "Built-in Chain",
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in built-in chain query: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "query_time": 0,
            "method": "Built-in Chain",
            "success": False
        }

def query_rag_with_custom_context(question):
    """Query using custom context handling"""
    try:
        start_time = time.time()
        
        # Retrieve documents
        retrieved_docs = vectorstore.similarity_search(question, k=6)
        doc_contents = [doc.page_content for doc in retrieved_docs]
        trimmed_context = truncate_context(doc_contents, max_tokens=1024)
        
        # Create prompt template
        template = """Use the following context to answer the question concisely.
        If you don't know, say "I don't know."

        Context: {context}

        Question: {question}
        """
        
        prompt = template.format(context=trimmed_context, question=question)
        response = llm.invoke(prompt)
        
        end_time = time.time()
        
        return {
            "answer": response,
            "query_time": round(end_time - start_time, 4),
            "method": "Custom Context",
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in custom context query: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "query_time": 0,
            "method": "Custom Context",
            "success": False
        }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        method = data.get('method', 'builtin')  # 'builtin' or 'custom'
        
        if not question:
            return jsonify({
                "error": "Question is required",
                "success": False
            }), 400
        
        # Check if RAG system is initialized
        if not vectorstore or not qa_chain or not llm:
            return jsonify({
                "error": "RAG system not initialized. Please check your configuration.",
                "success": False
            }), 500
        
        # Query based on selected method
        if method == 'custom':
            result = query_rag_with_custom_context(question)
        else:
            result = query_rag_with_built_in_chain(question)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    rag_status = "initialized" if (vectorstore and qa_chain and llm) else "not initialized"
    return jsonify({
        "status": "healthy",
        "rag_system": rag_status,
        "timestamp": time.time()
    })

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize RAG system on startup
    if initialize_rag():
        print("🚀 RAG system initialized successfully!")
        print("🌐 Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize RAG system. Please check your configuration.")
        print("Make sure you have:")
        print("1. GOOGLE_API_KEY in your .env file")
        print("2. python_documents.md file in the same directory")