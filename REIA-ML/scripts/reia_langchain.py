import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_openai.chat_models import ChatOpenAI

class REIA_langchain_RAG:
    def __init__(self):
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
        os.environ['LANGCHAIN_API_KEY'] = 'YOUR API KEY'  # Set your LangChain API key
        os.environ['LANGCHAIN_PROJECT'] = "REIA"
        os.environ['OPENAI_API_KEY'] = 'YOUR API KEY'  # Set your OpenAI API key
        os.environ['KMP_DUPLICATE_LIB_OK']='True'

        self.db_list = []
        self.tools = []
        self.agent_executor = None
        self.generated_data = False  # Initialize generated_data attribute


    def pdf_loader(self, file_path):
        try:
            pdfloader = PyPDFLoader(file_path)
            pdfpages = pdfloader.load_and_split()

            mydocuments = pdfloader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(mydocuments)
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(texts, embeddings)
            db.name = os.path.basename(file_path)  # Set the name attribute to the base name of the file
            self.db_list.append(db)
            return self  # Return self for method chaining
        except Exception as e:
            print(f"Error loading PDF from {file_path}: {e}")
            return self

    def initialize_agent(self):
        # Create retrievers for each database in db_list and add to tools
        for db in self.db_list:
            retriever = db.as_retriever()
            sanitized_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in db.name)
            tool = create_retriever_tool(
                retriever,
                f"{sanitized_name}_retriever",
                f"searches and returns documents regarding the {sanitized_name} system"
            )
            self.tools.append(tool)

        # Generate data for conversational agent
        llm = ChatOpenAI(temperature=0)
        self.agent_executor = create_conversational_retrieval_agent(llm, self.tools, verbose=True)
        self.generated_data = True  # Set flag to True after generating data


    def query(self, input_text):
        if self.agent_executor is None:
            raise RuntimeError("Agent executor not initialized. Call generate_data() first.")
        result = self.agent_executor.invoke({"input": input_text})
        return result


