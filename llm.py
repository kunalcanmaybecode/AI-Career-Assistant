from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from retriever import build_context_string

load_dotenv()


def analyze_skill_gap(resume_docs : List[Document],
                      jd_docs : List[Document],
                      query:str = "Analyze my Resume against the Job Description. Identify missing skills and provide a specific, actionable learning plan.",
                      chat_history: List[Tuple[str,str]] = None) -> str:

    if chat_history is None:
        chat_history = []
    
    resume_text = build_context_string(resume_docs)
    jd_text = build_context_string(jd_docs)


    system_instruction = """
    You are an expert technical career coach. 
    Analyze the provided Job Description (JD) and the candidate's Resume.
    1. Identify up to 3 critical skill gaps.
    2. Suggest a brief, high-level action plan in bullet points.
    
    CRITICAL INSTRUCTION: Keep your response concise, direct, and strictly under 200 words. Do not provide week-by-week breakdowns.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    model = ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=500 
    )
    
    chain = prompt | model | StrOutputParser()
    return chain.stream({
        "jd": jd_text,
        "resume": resume_text,
        "chat_history" : chat_history,
        "question" : query
        })


def generate_interview_questions(resume_docs: List[Document], 
                                jd_docs: List[Document],
                                query: str = "Hello, I am ready to begin the mock interview.",
                                chat_history: List[Tuple[str, str]] = None                                       
                                ) -> str:

    if chat_history is None:
        chat_history = []
    
    resume_text = build_context_string(resume_docs)
    jd_text = build_context_string(jd_docs)
    
    system_instruction = """
    You are a strict technical hiring manager conducting a mock interview.
    The human you are speaking to is the job candidate.
    Base your questions on the provided Job Description and Candidate Resume.
    
    CRITICAL RULES:
    1. Ask only ONE concise question at a time.
    2. If the candidate answers correctly, briefly validate it.
    3. If the candidate says "I don't know", asks for the answer, or answers incorrectly, briefly explain the correct answer in 1-2 sentences. 
    4. AFTER evaluating their answer, ALWAYS immediately ask a NEW interview question.
    5. Never break character. Keep your total response under 4 sentences.
    
    Job Description:
    {jd}
    
    Candidate Resume:
    {resume}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        (MessagesPlaceholder(variable_name="chat_history")),
        ("human", "{question}")
    ])
    
    model = ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,
        max_tokens = 500
    )
    
    chain = prompt | model | StrOutputParser()
    return chain.stream({"jd" : jd_text, 
                        "resume" : resume_text,
                        "chat_history" : chat_history,
                        "question" : query
                        })
