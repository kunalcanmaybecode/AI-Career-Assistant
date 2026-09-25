# AI Career Assistant (RAG-Powered)

An AI-powered technical interview preparation and career coaching tool. This application uses a Retrieval-Augmented Generation (RAG) pipeline to analyze a user's Resume against a specific Job Description (JD) to provide highly contextual, actionable feedback and conduct dynamic mock interviews.

## Features

*   **Dual-Agent Architecture:**
    *   **Career Coach Mode:** Performs a gap analysis between your resume and the JD, identifying missing skills and generating a concise, actionable learning plan.
    *   **Mock Interviewer Mode:** Acts as a strict technical hiring manager, asking one question at a time based on your specific experience and the role's requirements, grading your answers in real-time.
*   **Intelligent Document Retrieval (RAG):** Uses local vector embeddings to search through uploaded PDFs, ensuring the AI only asks questions relevant to the provided job description and candidate history.
*   **Streaming UI:** Real-time typewriter effect for instant AI responses, built with Streamlit.
*   **Optimized Inference:** Implements session-state caching for heavy embedding models and CrossEncoder rerankers to ensure instant chat responsiveness without memory overhead.

## Tech Stack

*   **Frontend UI:** Streamlit
*   **Orchestration:** LangChain (LCEL)
*   **LLM:** Groq API (Qwen/Llama 3) for high-speed, low-latency inference
*   **Vector Database:** ChromaDB (Persistent local SQLite storage)
*   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
*   **Reranking:** Sentence-Transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
*   **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

## How It Works (Architecture)

1.  **Ingestion:** The user uploads a Resume and JD (PDFs). The app chunks the text, applies document-type metadata, generates vector embeddings, and stores them in a local ChromaDB collection.
2.  **Retrieval:** When a user sends a message, the app queries ChromaDB for the top *K* most relevant chunks, applying a metadata filter to isolate Resume data from JD data.
3.  **Reranking:** A CrossEncoder evaluates the retrieved chunks against the user's prompt to surface the highest-quality context.
4.  **Generation:** The context and conversation history are passed into a LangChain prompt template and streamed via Groq's API back to the Streamlit UI.
