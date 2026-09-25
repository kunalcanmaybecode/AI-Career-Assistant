from ingestion import load_existing_db
from retriever import retrieve_context
from llm import analyze_skill_gap, generate_interview_questions

def main():
    db = load_existing_db()
    
    print("Welcome to the AI Career Assistant!")
    print("1. Career Coach (Skill Gap Analysis & Advice)")
    print("2. Mock Interviewer")
    choice = input("Select a mode (1 or 2): ")

    chat_history = []

    if choice == "1":
        print("\n" + "="*50)
        print("COACH MODE: Generating initial Skill Gap Analysis...")
        print("="*50 + "\n")
        
        resume_docs = retrieve_context(db, "core skills and experience", k=5, doc_type_filter="resume")
        jd_docs = retrieve_context(db, "job requirements and qualifications", k=5, doc_type_filter="jd")
        
        initial_response = analyze_skill_gap(resume_docs, jd_docs)
        print(f"Coach: {initial_response}\n")
        
        initial_query = "Analyze my Resume against the Job Description. Identify missing skills and provide a specific, actionable learning plan."
        chat_history.append(("human", initial_query))
        chat_history.append(("ai", initial_response))
        
        while True:
            user_input = input("You (Type 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            res_docs = retrieve_context(db, user_input, k=3, doc_type_filter="resume")
            j_docs = retrieve_context(db, user_input, k=3, doc_type_filter="jd")
            
            response = analyze_skill_gap(
                resume_docs=res_docs, 
                jd_docs=j_docs, 
                query=user_input, 
                chat_history=chat_history
            )
            print(f"\nCoach: {response}\n")
            
            chat_history.append(("human", user_input))
            chat_history.append(("ai", response))

    elif choice == "2":
        print("\n" + "="*50)
        print("INTERVIEW MODE: Preparing first question...")
        print("="*50 + "\n")
        
        resume_docs = retrieve_context(db, "projects and technical experience", k=5, doc_type_filter="resume")
        jd_docs = retrieve_context(db, "technical requirements", k=5, doc_type_filter="jd")
        
        initial_response = generate_interview_questions(resume_docs, jd_docs)
        print(f"Interviewer: {initial_response}\n")
        
        initial_query = "Hello, I am ready to begin the mock interview."
        chat_history.append(("human", initial_query))
        chat_history.append(("ai", initial_response))
        
        while True:
            user_input = input("You (Type 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            res_docs = retrieve_context(db, user_input, k=3, doc_type_filter="resume")
            j_docs = retrieve_context(db, user_input, k=3, doc_type_filter="jd")
            
            response = generate_interview_questions(
                resume_docs=res_docs, 
                jd_docs=j_docs, 
                query=user_input, 
                chat_history=chat_history
            )
            print(f"\nInterviewer: {response}\n")
            
            chat_history.append(("human", user_input))
            chat_history.append(("ai", response))
            
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
