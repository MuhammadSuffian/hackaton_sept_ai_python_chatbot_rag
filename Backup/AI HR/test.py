import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from groq import Groq
import PyPDF2
import io
import re

# Initialize Groq client with environment variable
import os
groq_client = Groq(api_key="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")

# Remove hidden reasoning from model outputs
THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def hide_thinking(text: str) -> str:
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = THINK_TAG_RE.sub("", text)
    drop_prefixes = ("reasoning:", "thought:", "chain-of-thought:", "scratchpad:")
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith(drop_prefixes)]
    return "\n".join(lines).strip()

# Streamlit UI
st.title("🔍 RAG Chatbot with Groq + Streamlit")

# App state
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "num_docs" not in st.session_state:
    st.session_state["num_docs"] = 0
if "num_chunks" not in st.session_state:
    st.session_state["num_chunks"] = 0

# Step 1: Upload document(s)
uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    st.info("Files selected. Click 'Process documents' to build the index.")
    if st.button("Process documents"):
        with st.spinner("Processing and indexing documents..."):
            try:
                docs = []
                for file in uploaded_files:
                    if file.type == "text/plain":
                        text = file.read().decode("utf-8", errors="ignore")
                    elif file.type == "application/pdf":
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                        text = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text() or ""
                            text += page_text + "\n"
                    else:
                        st.error(f"Unsupported file type: {file.type}")
                        continue

                    text = (text or "").strip()
                    if not text:
                        st.warning(f"No extractable text in {file.name}; skipping.")
                        continue

                    docs.append(Document(page_content=text, metadata={"filename": file.name}))

                if not docs:
                    st.error("No valid documents to index.")
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(docs)

                    if not chunks:
                        st.error("Documents produced zero chunks; try different files.")
                    else:
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        st.session_state["vectorstore"] = FAISS.from_documents(chunks, embeddings)
                        st.session_state["num_docs"] = len(docs)
                        st.session_state["num_chunks"] = len(chunks)
                        st.success(f"✅ Indexed {len(docs)} documents into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# Query UI (enabled once an index exists)
vectorstore_ready = st.session_state.get("vectorstore") is not None
query = st.text_input("Ask a question about your documents:", disabled=not vectorstore_ready)

if query and vectorstore_ready:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            retrieved_docs = st.session_state["vectorstore"].similarity_search(query, k=3)
            if not retrieved_docs:
                st.warning("No relevant context found in the indexed documents.")
            else:
                context = "\n".join([d.page_content for d in retrieved_docs])
                prompt = f"""Instruction: Provide only the final answer. Do not include chain-of-thought, hidden reasoning, or <think> blocks.
                Answer the question using only the context below:
                Context: {context}
                Question: {query}
                """

                response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content
                answer = hide_thinking(answer)
                st.subheader("💡 Answer")
                st.write(answer)

                with st.expander("Show retrieved context"):
                    for i, d in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Chunk {i}** — {d.metadata.get('filename', 'unknown')}")
                        st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))
        except Exception as e:
            st.error(f"Query failed: {e}")

# Sidebar status
with st.sidebar:
    st.markdown("**Index status**")
    if vectorstore_ready:
        st.success(f"Ready: {st.session_state['num_docs']} docs / {st.session_state['num_chunks']} chunks")
    else:
        st.info("No index yet. Upload files and click Process.")
