import PyPDF2
import gradio as gr
import ollama
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Paths
CHROMA_PATH = "chroma_db"
DEFAULT_PDF_PATH = "data/policies_kb.pdf"

# Load embedding model
embedding_function = HuggingFaceEmbeddings()


db = None

def extract_text_pypdf2(pdf_path):
   
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise

def process_and_store_pdf():
 
    try:
        print(f"Processing PDF: {DEFAULT_PDF_PATH}")
        text = extract_text_pypdf2(DEFAULT_PDF_PATH)
        if not text.strip():
            raise ValueError("PDF appears to be empty or unreadable.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_text(text)
        print(f"Split PDF into {len(chunks)} chunks")

        db_local = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        docs = [Document(page_content=chunk, metadata={"source": DEFAULT_PDF_PATH}) for chunk in chunks]
        db_local.add_documents(docs)
        db_local.persist()
        print("PDF processed and stored successfully!")
        return db_local
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

def ask_pdf_question(query):
   
    try:
        print("Retrieving relevant chunks...")
        results = db.similarity_search_with_score(query, k=5)

        if not results or all(score > 0.8 for _, score in results):
            log_escalated_query(query)
            return "⚠️ I'm not confident in answering this. Please contact our insurance advisor at support@insureai.com."

        context = "\n\n---\n\n".join([doc[0].page_content for doc in results])

        prompt = f"Context:\n{context}\n\nUser: {query}\nAI:"
        print("Querying Ollama...")

       
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        answer = response.get("message", {}).get("content", "").strip()

        if not answer:
            return "⚠️ Got an empty response. Try again."

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in ["i'm not sure", "not enough information", "cannot help"]):
            log_escalated_query(query)
            return "🤖 This seems like a complex case. Transferring to a human expert: support@tcsion.com."

        return answer
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"⚠️ Error generating response: {str(e)}"

def log_escalated_query(query):
   
    try:
        with open("escalated_queries.txt", "a") as f:
            f.write(query + "\n")
        print(f"Logged escalated query: {query}")
    except Exception as e:
        print(f"Error logging query: {str(e)}")


print("Initializing and processing internal PDF...")
db = process_and_store_pdf()

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("##  Chat with Insurance policy agent\n_Type your question below and get AI-powered answers based on insurance policies._")

    with gr.Row():
        query_input = gr.Textbox(label="Ask a Question", placeholder="e.g., What is the claim process for theft?")
        query_button = gr.Button("Get Answer")

    output_text = gr.Textbox(label="Agent's Response", lines=5)

    query_button.click(fn=ask_pdf_question, inputs=query_input, outputs=output_text)

if __name__ == "__main__":
    print("Launching Gradio app...")
    app.launch(share=True)
