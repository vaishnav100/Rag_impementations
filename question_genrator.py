from flask import Flask, render_template, request, jsonify, session
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
import random
from werkzeug.utils import secure_filename
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure session
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'md', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class DocumentSession:
    """Class to manage document session state"""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None
        self.current_document_content = ""
        self.generated_questions = []
        self.asked_questions = []
        self.created_at = datetime.now()
    
    def reset(self):
        """Reset all session data"""
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None
        self.current_document_content = ""
        self.generated_questions = []
        self.asked_questions = []
        self.created_at = datetime.now()

def get_session_data():
    """Get or create session data"""
    if 'session_id' not in session:
        session['session_id'] = str(random.randint(100000, 999999))
    
    session_id = session['session_id']
    
    # Use app context to store session data
    if not hasattr(app, 'sessions'):
        app.sessions = {}
    
    if session_id not in app.sessions:
        app.sessions[session_id] = DocumentSession()
    
    return app.sessions[session_id]

def cleanup_old_sessions():
    """Clean up old sessions (optional, for memory management)"""
    if hasattr(app, 'sessions'):
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, doc_session in app.sessions.items():
            # Remove sessions older than 1 hour
            if (current_time - doc_session.created_at).seconds > 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del app.sessions[session_id]

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_document_file(filepath):
    """Read different types of document files"""
    try:
        file_extension = filepath.rsplit('.', 1)[1].lower()
        
        if file_extension in ['txt', 'md']:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == 'pdf':
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                logger.error("PyPDF2 not installed. Cannot read PDF files.")
                return None
        elif file_extension == 'docx':
            try:
                from docx import Document
                doc = Document(filepath)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                logger.error("python-docx not installed. Cannot read DOCX files.")
                return None
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        logger.error(f"Error reading document: {str(e)}")
        return None

def initialize_rag_from_content(content, doc_session):
    """Initialize the RAG system from document content"""
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

        # Split document content into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create documents from content
        texts = text_splitter.create_documents([content])
        
        if not texts:
            raise ValueError("No text chunks created from document")

        # Create FAISS vector store
        doc_session.vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)

        # Create prompt template for Q&A
        qa_template = """Use the following context to answer the question concisely and accurately.
        If you don't know the answer based on the context, say "I don't know based on the provided document."

        Context: {context}

        Question: {question}
        
        Answer:"""

        QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

        # Initialize LLM
        doc_session.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.1
        )

        # Create RetrievalQA chain
        doc_session.qa_chain = RetrievalQA.from_chain_type(
            llm=doc_session.llm,
            chain_type="stuff",
            retriever=doc_session.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        doc_session.current_document_content = content
        logger.info("RAG system initialized successfully from uploaded document")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

def generate_questions_from_document(content, llm, num_questions=3, difficulty="mixed"):
    """
    Generate questions with two answer options from document content.
    Returns a list of dicts: [{ "question": str, "options": [str, str] }, ...]
    """
    try:
        if not llm:
            raise ValueError("LLM not initialized")

        question_prompt = f"""Based on the following document content, generate {num_questions} thoughtful questions that test understanding of the material.

            Requirements:
            1. Questions should be diverse and cover different aspects of the document.
            2. Include a mix of factual, conceptual, and analytical questions.
            3. Each question must be followed by two plausible answer options labeled A) and B).
            4. Ensure one of the options is clearly correct based on the document.
            5. Questions and options should be clear, specific, and answerable from the document.
            6. Return ONLY the questions and their options, each question on a new line, numbered (1., 2., 3.).

            Example format:
            1. What is the main purpose of X?
            A) Reason 1
            B) Reason 2

            Document content:
            {content[:3000]}...

            Generate {num_questions} questions with two options each:"""

        # Call the LLM with the prompt
        response = llm.invoke(question_prompt)

        questions = []
        current_q = {}
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Match question lines starting with number dot e.g. "1."
            if re.match(r"^\d+\.", line):
                # If a question is already being built, save it first
                if current_q:
                    questions.append(current_q)
                    current_q = {}

                # Extract the question text after number and dot
                question_text = line.split('.', 1)[1].strip()
                current_q["question"] = question_text
                current_q["options"] = []

            # Match answer option lines starting with A) or B)
            elif line.startswith("A)") or line.startswith("B)"):
                current_q.setdefault("options", []).append(line)

        # Add the last question if any
        if current_q:
            questions.append(current_q)

        # Return only the requested number of questions
        return questions[:num_questions]

    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        # Return fallback question(s) in same structure
        return [{
            "question": "What are the main topics discussed in this document?",
            "options": ["A) Introduction and summary", "B) Irrelevant content"]
        }]

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index2.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Handle document upload and initialize RAG system"""
    try:
        # Clean up old sessions periodically
        cleanup_old_sessions()
        
        # Get current session data
        doc_session = get_session_data()
        
        if 'document' not in request.files:
            return jsonify({"error": "No file uploaded", "success": False}), 400
        
        file = request.files['document']
        if file.filename == '':
            return jsonify({"error": "No file selected", "success": False}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read document content
            content = read_document_file(filepath)
            if not content:
                return jsonify({"error": "Could not read document content", "success": False}), 400
            
            # Initialize RAG system
            if not initialize_rag_from_content(content, doc_session):
                return jsonify({"error": "Failed to initialize RAG system", "success": False}), 500
            
            # Generate initial questions
            doc_session.generated_questions = generate_questions_from_document(
                content, doc_session.llm, num_questions=3
            )
            doc_session.asked_questions = []
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                "success": True,
                "message": "Document uploaded and processed successfully",
                "questions": doc_session.generated_questions,
                "document_length": len(content)
            })
        else:
            return jsonify({"error": "Invalid file type", "success": False}), 400
            
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return jsonify({"error": f"Upload error: {str(e)}", "success": False}), 500

@app.route('/api/answer', methods=['POST'])
def answer_question():
    try:
        # Get current session data
        doc_session = get_session_data()
        
        data = request.get_json()
        print("Received data:", data)

        question_data = data.get('question', {})
        question_text = question_data.get('question', '').strip()

        if not question_text:
            return jsonify({"error": "Question is required", "success": False}), 400

        if not doc_session.qa_chain:
            return jsonify({"error": "No document loaded", "success": False}), 400

        # Get answer using RAG
        start_time = time.time()
        results = doc_session.qa_chain({"query": question_text})
        end_time = time.time()

        if question_text not in doc_session.asked_questions:
            doc_session.asked_questions.append(question_text)

        return jsonify({
            "success": True,
            "question": question_text,
            "answer": results["result"],
            "query_time": round(end_time - start_time, 4),
            "questions_answered": len(doc_session.asked_questions)
        })

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}", "success": False}), 500

@app.route('/api/generate-more-questions', methods=['POST'])
def generate_more_questions():
    """Generate more questions from the document"""
    try:
        # Get current session data
        doc_session = get_session_data()
        
        if not doc_session.current_document_content:
            return jsonify({"error": "No document loaded", "success": False}), 400
        
        # Generate new questions
        new_questions = generate_questions_from_document(
            doc_session.current_document_content, doc_session.llm, num_questions=3
        )
        
        # Filter out questions that are too similar to previously asked ones
        filtered_questions = []
        for q in new_questions:
            is_similar = False
            q_text = q['question'] if isinstance(q, dict) else q
            for asked_q in doc_session.asked_questions:
                asked_q_text = asked_q['question'] if isinstance(asked_q, dict) else asked_q
                if len(set(q_text.lower().split()) & set(asked_q_text.lower().split())) > len(q_text.split()) * 0.5:
                    is_similar = True
                    break
            if not is_similar:
                filtered_questions.append(q)
        
        # If we don't have enough unique questions, generate more
        if len(filtered_questions) < 3:
            additional_questions = generate_questions_from_document(
                doc_session.current_document_content, 
                doc_session.llm,
                num_questions=5
            )
            for q in additional_questions:
                if q not in filtered_questions and len(filtered_questions) < 3:
                    filtered_questions.append(q)
        
        doc_session.generated_questions.extend(filtered_questions[:3])
        
        return jsonify({
            "success": True,
            "questions": filtered_questions[:3],
            "total_questions_generated": len(doc_session.generated_questions),
            "questions_answered": len(doc_session.asked_questions)
        })
        
    except Exception as e:
        logger.error(f"Error generating more questions: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}", "success": False}), 500

@app.route('/api/custom-question', methods=['POST'])
def custom_question():
    """Handle custom user questions about the document"""
    try:
        # Get current session data
        doc_session = get_session_data()
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Question is required", "success": False}), 400
        
        if not doc_session.qa_chain:
            return jsonify({"error": "No document loaded", "success": False}), 400
        
        # Get answer using RAG
        start_time = time.time()
        results = doc_session.qa_chain({"query": question})
        end_time = time.time()
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": results["result"],
            "query_time": round(end_time - start_time, 4),
            "is_custom": True
        })
        
    except Exception as e:
        logger.error(f"Error answering custom question: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}", "success": False}), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the current session"""
    try:
        # Get current session data and reset it
        doc_session = get_session_data()
        doc_session.reset()
        
        return jsonify({
            "success": True,
            "message": "Session reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}", "success": False}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current session status"""
    try:
        # Get current session data
        doc_session = get_session_data()
        
        return jsonify({
            "success": True,
            "document_loaded": bool(doc_session.current_document_content),
            "questions_generated": len(doc_session.generated_questions),
            "questions_answered": len(doc_session.asked_questions),
            "rag_initialized": bool(doc_session.qa_chain)
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}", "success": False}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Document Q&A Learning System")
    print("📚 Upload a document and get interactive questions!")
    print("🌐 Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)