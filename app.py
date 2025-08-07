import json
import re
import os
import shutil
import uuid
import requests
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, validator
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import uvicorn
import tempfile
import time
import logging
from concurrent.futures import ThreadPoolExecutor

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(name)s')

# --- Pydantic Models ---
class ParsedQuery(BaseModel):
    age: Optional[int] = Field(None, description="The age of the claimant in years.")
    procedure: Optional[str] = Field(None, description="The medical procedure mentioned in the query (e.g., knee surgery).")
    location: Optional[str] = Field(None, description="The geographical location where the procedure occurred (e.g., Pune).")
    policy_duration_months: Optional[int] = Field(None, description="The duration of the insurance policy in months.")

    @validator('age', 'policy_duration_months', pre=True)
    def parse_numeric(cls, v):
        if isinstance(v, str):
            v = v.strip().lower().replace('years old', '').replace('months', '').replace('old', '')
            if v.isdigit():
                return int(v)
        return v

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The approval status of the claim. Must be 'approved', 'rejected', or 'further review'.")
    amount: str = Field(..., description="The coverage amount, if applicable. Use 'N/A' if not specified.")
    justification: str = Field(..., description="A detailed explanation for the decision, referencing the specific clause identifiers from the policy documents.")

class RunRequest(BaseModel):
    documents: List[str]
    questions: List[str]

# --- LLM and Prompts ---
OLLAMA_PARSING_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"
OLLAMA_REASONING_MODEL = "mistral:7b-instruct-v0.2-q4_K_M"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

llm_parser = None
llm_reasoning = None
embeddings_model = None

# --- FastAPI App Initialization (CORRECTED POSITION) ---
app = FastAPI(
    title="LLM-Powered Insurance Query System",
    description="A fast API for processing pre-indexed documents.",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    global llm_parser, llm_reasoning, embeddings_model
    logging.info("Initializing LLM and embedding models...")
    llm_parser = Ollama(model=OLLAMA_PARSING_MODEL)
    llm_reasoning = Ollama(model=OLLAMA_REASONING_MODEL)
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    logging.info("Models loaded and ready!")

parsing_prompt_template = PromptTemplate(
    template="You are a data extraction expert. Your goal is to extract structured information from a user's query about an insurance claim. Follow these steps precisely to ensure accuracy:\n\n"
             "1. First, read the User Query and identify all key entities: age, procedure, location, and policy duration.\n"
             "2. For each identified entity, consider its context in the query.\n"
             "3. If policy duration is in years, convert it to months (e.g., '2 years' becomes 24).\n"
             "4. If a field is not present in the query, mark it as null.\n"
             "5. Based on your reasoning, output a single JSON object with the extracted data. Ensure the JSON format is perfect, with no extra text or explanations outside of the JSON block.\n\n"
             "User Query: {query}\n"
             "Format Instructions: {format_instructions}\n"
             "Reasoning and Final JSON:",
    input_variables=["query"],
    partial_variables={"format_instructions": JsonOutputParser(pydantic_object=ParsedQuery).get_format_instructions()},
)

QA_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are a helpful Q&A assistant. Your task is to provide a concise, one- or two-sentence answer to the user's question based *strictly* on the provided context.
    Do not provide a decision, amount, or justification. Just provide the answer.

    Context:
    {retrieved_context}

    Question: {query}

    Concise Answer:
    """,
    input_variables=["retrieved_context", "query"]
)

# --- Core Functions ---
def parse_query(user_query: str) -> Optional[ParsedQuery]:
    age = None
    age_match = re.search(r'(\d+)\s*[-_]*\s*year[s]?', user_query, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
    policy_duration_months = None
    months_match = re.search(r'(\d+)\s*[-_]*\s*month[s]?', user_query, re.IGNORECASE)
    if months_match:
        policy_duration_months = int(months_match.group(1))
    else:
        years_match = re.search(r'(\d+)\s*[-_]*\s*year[s]?', user_query, re.IGNORECASE)
        if years_match:
            policy_duration_months = int(years_match.group(1)) * 12
    try:
        parsing_chain = parsing_prompt_template | llm_parser | JsonOutputParser(pydantic_object=ParsedQuery)
        llm_extracted_data = parsing_chain.invoke({"query": user_query})
        if age is not None:
            llm_extracted_data['age'] = age
        if policy_duration_months is not None:
            llm_extracted_data['policy_duration_months'] = policy_duration_months
        return ParsedQuery.model_validate(llm_extracted_data)
    except Exception as e:
        print(f"Error parsing query: {e}")
        return None

def load_and_chunk_docs_from_urls(urls: List[str]) -> List:
    all_chunks = []
    for url in urls:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                shutil.copyfileobj(response.raw, temp_file)
                temp_file_path = temp_file.name

            documents = PyPDFLoader(temp_file_path).load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
            chunks = text_splitter.split_documents(documents)
            
            refined_chunks = []
            for i, chunk in enumerate(chunks):
                content = re.sub(r'\n{2,}', '\n', chunk.page_content).strip()
                content = re.sub(r'Page \d+', '', content)
                chunk.page_content = content
                chunk.metadata['chunk_id'] = f"chunk_{i}_{str(uuid.uuid4())[:8]}"
                clause_match = re.search(r'(Clause|Section)\s+[\d\.]+', content, re.IGNORECASE)
                if clause_match:
                    chunk.metadata['clause_identifier'] = clause_match.group(0)
                else:
                    chunk.metadata['clause_identifier'] = f"source - page {chunk.metadata.get('page', 'N/A')}"
                chunk.metadata['document_source'] = url
                refined_chunks.append(chunk)
            all_chunks.extend(refined_chunks)

        except Exception as e:
            logging.error(f"Failed to process document from URL {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing document from URL: {url}. Details: {str(e)}")
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return all_chunks

def run_main_qa_pipeline(user_query: str, chunks: List) -> str:
    try:
        vector_db = Chroma.from_documents(chunks, embeddings_model)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(user_query)
        retrieved_context = "\n\n".join([f"--- Clause from {doc.metadata.get('document_source')} - {doc.metadata.get('clause_identifier')} ---\n{doc.page_content}" for doc in retrieved_docs])
    except Exception as e:
        logging.error(f"Error during retrieval or vector_db creation: {e}")
        return f"Error in retrieving policy clauses: {str(e)}"

    try:
        final_prompt = QA_PROMPT_TEMPLATE.format(
            query=user_query,
            retrieved_context=retrieved_context,
        )
        raw_output = llm_reasoning.invoke(final_prompt)
        return raw_output.strip()
    except Exception as e:
        logging.error(f"Error during LLM reasoning: {e}")
        return f"An error occurred during LLM reasoning: {str(e)}"

# --- New Endpoint: The API for the RAG pipeline ---
@app.post("/api/v1/hackrx/run", tags=["Query Processing"])
async def run_rag(
    request_body: RunRequest
):
    start_time = time.time()
    
    if not request_body.documents or not request_body.questions:
        raise HTTPException(status_code=400, detail="Documents and questions cannot be empty.")
    
    try:
        logging.info(f"Processing request for {len(request_body.questions)} questions...")
        chunks = load_and_chunk_docs_from_urls(request_body.documents)
        
        all_answers = []
        for question in request_body.questions:
            answer = run_main_qa_pipeline(question, chunks)
            all_answers.append(answer)
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

    end_time = time.time()
    response_time = end_time - start_time
    logging.info(f"Request completed in {response_time:.2f} seconds.")

    return {"answers": all_answers}

if __name__ == "__main__":

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
