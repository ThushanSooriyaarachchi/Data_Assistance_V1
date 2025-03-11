from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import os
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="Ollama RAG API")

class Query(BaseModel):
    question: str

# Initialize the RAG pipeline
def initialize_rag():
    # Load the PDF
    pdf_path = os.path.join("data", "Job Description.pdf")
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()

    # Split into chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings()

    # Create the vector store and fill it with embeddings
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define llm
    llm = Ollama(model="mistral", base_url="http://ollama:11434")

    # Define the prompt
    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3,4 sentences.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

    llm_chain = LLMChain(
                      llm=llm, 
                      prompt=QA_CHAIN_PROMPT, 
                      callbacks=None, 
                      verbose=True)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
                      llm_chain=llm_chain,
                      document_variable_name="context",
                      document_prompt=document_prompt,
                      callbacks=None)
                  
    qa = RetrievalQA(
                      combine_documents_chain=combine_documents_chain,
                      verbose=True,
                      retriever=retriever,
                      return_source_documents=True)
    
    return qa

# Initialize the RAG system
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global qa_chain
    print("Initializing RAG system...")
    qa_chain = initialize_rag()
    print("RAG system initialized!")

@app.get("/")
def read_root():
    return {"message": "Ollama RAG API is running"}

@app.post("/query")
def query(query: Query):
    result = qa_chain(query.question)
    return {
        "answer": result['result'],
        "source_documents": [
            {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
            for doc in result['source_documents']
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)