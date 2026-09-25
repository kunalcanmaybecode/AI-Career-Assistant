import streamlit as st
import os
from ingestion import load_existing_db, ingest_documents
from retriever import retrieve_context
from llm import analyze_skill_gap, generate_interview_questions

st.set_page_config(page_title="AI Career Assistant", layout="wide")
st.title("AI Career Assistant")

@st.cache_resource
def get_db():
    return load_existing_db()

db = get_db()

with st.sidebar:
    st.header("1. Upload Documents")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    
    if st.button("Process Documents"):
        if resume_file and jd_file:
            with st.spinner("Chunking and saving to ChromaDB..."):
                with open("temp_resume.pdf", "wb") as f:
                    f.write(resume_file.read())
                with open("temp_jd.pdf", "wb") as f:
                    f.write(jd_file.read())
                
                try:
                    db.delete_collection()
                except Exception:
                    pass
                
                files_dict = {
                    "resume": "temp_resume.pdf",
                    "jd": "temp_jd.pdf"
                }
                ingest_documents(files_dict, reset_db=False)
                
                st.cache_resource.clear()
                st.session_state.messages = []
                st.success("Database updated! Ready to chat.")
                st.rerun() 
        else:
            st.error("Please upload both a Resume and a JD.")

    st.divider()
    
    st.header("2. Choose Mode")
    mode = st.sidebar.radio("Select an agent:", ["Career Coach", "Mock Interviewer"])
    
    if st.button("Restart Conversation"):
        st.session_state.messages = []
        st.rerun()

if "current_mode" not in st.session_state:
    st.session_state.current_mode = mode

if st.session_state.current_mode != mode:
    st.session_state.messages = []  
    st.session_state.current_mode = mode

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if len(st.session_state.messages) == 0:
    
    with st.spinner(f"Initializing {mode}..."):
        res_docs = retrieve_context(db, "core skills and experience", k=5, doc_type_filter="resume")
        jd_docs = retrieve_context(db, "job requirements and qualifications", k=5, doc_type_filter="jd")
    
    if not res_docs and not jd_docs:
        welcome_msg = f"Welcome to the {mode}! Please upload your Resume and a Job Description in the sidebar and click 'Process Documents' to begin."
        
        with st.chat_message("assistant"):
            st.markdown(welcome_msg)
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    else:
        if mode == "Career Coach":
            initial_query = "Analyze my Resume against the Job Description. Identify missing skills and provide a specific, actionable learning plan."
        elif mode == "Mock Interviewer":
            initial_query = "Hello, I am ready to begin the mock interview."
        
        with st.chat_message("user"):
            st.markdown(initial_query)
        st.session_state.messages.append({"role": "user", "content": initial_query})
        
        with st.chat_message("assistant"):
            if mode == "Career Coach":
                stream = analyze_skill_gap(res_docs, jd_docs, query=initial_query)
            else:
                stream = generate_interview_questions(res_docs, jd_docs, query=initial_query)
                
            initial_response = st.write_stream(stream)            
        st.session_state.messages.append({"role": "assistant", "content": initial_response})



if user_input := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    formatted_history = [
        ("human" if m["role"] == "user" else "ai", m["content"])
        for m in st.session_state.messages[:-1] 
    ]
    
    with st.chat_message("assistant"):
        res_docs = retrieve_context(db, user_input, k=3, doc_type_filter="resume")
        jd_docs = retrieve_context(db, user_input, k=3, doc_type_filter="jd")
        
        if mode == "Career Coach":
            stream = analyze_skill_gap(
                resume_docs=res_docs, 
                jd_docs=jd_docs, 
                query=user_input, 
                chat_history=formatted_history
            )
        else:
            stream = generate_interview_questions(
                resume_docs=res_docs, 
                jd_docs=jd_docs, 
                query=user_input, 
                chat_history=formatted_history
            )
        
        ai_response = st.write_stream(stream)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_response})