# main.py

from reia_langchain import REIA_langchain_RAG

rag = REIA_langchain_RAG()
rag.pdf_loader('../dataset/powerwall-overview-welcome-guide.pdf') \
    .pdf_loader('../dataset/solar_panels.pdf') \
    .initialize_agent() 


