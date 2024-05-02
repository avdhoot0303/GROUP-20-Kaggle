from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from main import REIA_langchain_RAG  # Import your LangChain class

# Load pre-trained model and tokenizer
try:
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    print("Error loading model:", e)
    raise

# Create FastAPI application
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (change this to your frontend URL)
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow specified HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Request model for chatbot endpoint
class ChatbotRequest(BaseModel):
    input_text: str

# Response model for chatbot endpoint
class ChatbotResponse(BaseModel):
    response: str

# Initialize LangChain instance
try:
    langchain = REIA_langchain_RAG()
except Exception as e:
    print("Error initializing LangChain:", e)
    raise

# Define endpoint for chatbot
@app.post("/api/chatbot", response_model=ChatbotResponse)
async def chatbot(request: ChatbotRequest):
    input_text = request.input_text
    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is required")
    
    # Use LangChain to generate response
    try:
        # Load PDF files and initialize agent if not already done
        if not langchain.generated_data:
            pdf_files = ['../dataset/powerwall-overview-welcome-guide.pdf', 
                         '../dataset/solar_panels.pdf']
            for file in pdf_files:
                langchain.pdf_loader(file)
            langchain.initialize_agent()

        result = langchain.query(input_text)
        response_text = result['output']
    except Exception as e:
        print("Error generating response with LangChain:", e)
        raise HTTPException(status_code=500, detail="Internal server error")
    
    return {"response": response_text}
