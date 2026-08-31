# rag_local_fixed.py
"""
Complete Local RAG System - Working with Latest LangChain
100% Free, No API Keys Required
"""

import os
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# Updated imports for latest LangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

print("=" * 60)
print("🏥 Local RAG Discharge Summary System (100% FREE)")
print("=" * 60)

# ============================================
# CREATE SAMPLE DOCUMENTS
# ============================================

def create_sample_documents():
    """Create sample clinical documents"""
    
    doc_dir = Path("./sample_docs")
    doc_dir.mkdir(exist_ok=True)
    
    # Complete patient record as a single document (simpler)
    patient_record = """
    ========================================
    PATIENT: John Smith
    MRN: 00-12-34-56
    Age: 68
    ========================================
    
    ADMISSION (March 15, 2026):
    ----------------------------------------
    CHIEF COMPLAINT: Shortness of breath for 3 days, fever, productive cough
    
    ADMISSION DIAGNOSIS: Community-Acquired Pneumonia, Right Lower Lobe
    
    VITALS: BP 145/88, HR 102, Temp 101.5°F (38.6°C), O2 sat 88% on room air
    
    PAST MEDICAL HISTORY: Hypertension, Type 2 Diabetes, Hyperlipidemia
    
    MEDICATIONS ON ADMISSION: Lisinopril 20mg daily, Metformin 1000mg twice daily, Atorvastatin 40mg daily
    
    ALLERGIES: Penicillin (hives and rash), Codeine (nausea)
    
    PHYSICAL EXAM: Crackles at right base, decreased breath sounds, egophony positive
    
    ========================================
    HOSPITAL COURSE
    ========================================
    
    DAY 2 (March 16, 2026):
    Patient feeling better. Shortness of breath improved.
    Vitals: BP 138/84, HR 96, Temp 100.5°F, O2 sat 94% on 2L NC
    Labs: WBC 11.8 (improving from 15.2), CRP 8.2
    Plan: Continue antibiotics, wean oxygen
    
    DAY 3 (March 17, 2026):
    Patient feeling almost normal. Mild dry cough only.
    Vitals: BP 125/78, HR 76, Temp 98.6°F, O2 sat 97% on room air
    Lungs: Clear to auscultation
    Labs: WBC 9.2 (normal), CRP 2.1
    Plan: Transition to oral antibiotics
    
    DAY 4 (March 18, 2026):
    Ready for discharge. No respiratory symptoms.
    Vitals: BP 122/76, HR 72, Temp 98.2°F, O2 sat 98% room air
    Plan: Discharge tomorrow
    
    ========================================
    LABORATORY RESULTS
    ========================================
    
    Admission Labs (March 15):
    - WBC: 15.2 (H)
    - CRP: 18.4 (H)
    - Procalcitonin: 2.5 (H)
    - Glucose: 165 (H)
    - Creatinine: 1.2
    
    Sputum Culture: Streptococcus pneumoniae
    Sensitivity: Susceptible to Ceftriaxone, Levofloxacin
    
    Day 2 Labs:
    - WBC: 11.8
    - CRP: 8.2
    
    Day 3 Labs:
    - WBC: 9.2
    - CRP: 2.1
    
    ========================================
    DISCHARGE SUMMARY (March 19, 2026)
    ========================================
    
    DISCHARGE DIAGNOSIS: Community-Acquired Pneumonia - Resolved
    
    DISCHARGE MEDICATIONS:
    1. Levofloxacin 750mg PO daily x 5 days (complete full course)
    2. Lisinopril 20mg PO daily (resume home dose)
    3. Metformin 1000mg PO twice daily (resume)
    4. Atorvastatin 40mg PO daily (resume)
    
    DISCHARGE INSTRUCTIONS:
    - Complete full course of Levofloxacin (5 days total)
    - Follow up with PCP Dr. James Wilson within 7 days
    - Repeat chest X-ray in 4-6 weeks
    - Return to ED if: fever returns, worsening cough, shortness of breath
    - Resume all home medications
    
    FOLLOW-UP APPOINTMENTS:
    - Primary Care: March 26, 2026
    - Pulmonary Clinic: April 15, 2026 (if symptoms persist)
    
    ALLERGIES: Penicillin (hives, rash) - AVOID
    """
    
    # Save as single file for simplicity
    filepath = doc_dir / "patient_complete_record.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(patient_record)
    print(f"✅ Created: {filepath}")
    
    return doc_dir

# ============================================
# LOCAL RAG SYSTEM (UPDATED)
# ============================================

class LocalRAG:
    def __init__(self):
        print("\n🔧 Initializing Local RAG System...")
        
        # Use HuggingFace embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ HuggingFace embeddings loaded")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise
        
        # Use Ollama for LLM
        try:
            self.llm = Ollama(
                model="phi3",
                temperature=0.3,
                base_url="http://localhost:11434"
            )
            print("✅ Ollama LLM connected (phi3)")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("   Please ensure Ollama is running")
            raise
        
        self.vectorstore = None
        self.retriever = None
        
    def ingest(self, file_path: str):
        """Ingest document into vector store"""
        print("\n📥 Ingesting document into vector database...")
        
        # Read document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks by sections (double newlines)
        chunks = []
        sections = content.split('\n\n')
        
        for i, section in enumerate(sections):
            if section.strip():
                chunk = Document(
                    page_content=section.strip(),
                    metadata={
                        "source": file_path,
                        "chunk_id": i,
                        "preview": section[:100]
                    }
                )
                chunks.append(chunk)
        
        print(f"  ✓ Created {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./local_rag_db"
        )
        
        # Create retriever (FIXED: using correct method for latest LangChain)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        print(f"✅ Vector store created with {len(chunks)} chunks")
        return True
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.retriever:
            return "Please ingest documents first"
        
        # FIXED: Use invoke instead of get_relevant_documents
        docs = self.retriever.invoke(question)
        
        # Build context from retrieved documents
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = PromptTemplate(
            template="""You are a medical assistant. Answer based ONLY on the following patient records.

PATIENT RECORDS:
{context}

QUESTION: {question}

Provide a concise, factual answer using only the information above.
If the answer is not in the records, say "Not documented in patient records."

ANSWER:""",
            input_variables=["context", "question"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context[:4000], "question": question})
        
        return answer
    
    def generate_discharge_summary(self) -> str:
        """Generate complete discharge summary"""
        if not self.retriever:
            return "Please ingest documents first"
        
        print("\n📄 Generating discharge summary...")
        
        # Retrieve all relevant information
        docs = self.retriever.invoke("complete patient record discharge summary medications followup")
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = PromptTemplate(
            template="""Generate a complete clinical discharge summary using ONLY the information below.

PATIENT RECORDS:
{context}

Create these sections:

ADMISSION DIAGNOSIS:
[Write the diagnosis from records]

DISCHARGE DIAGNOSIS:
[Write the final diagnosis]

HOSPITAL COURSE:
[Write chronological summary along with key vitals and lab markers]

DISCHARGE MEDICATIONS:
[List with doses and frequencies]

DISCHARGE INSTRUCTIONS/FOLLOW-UP:
[List instructions and appointments]

ALLERGIES:
[List allergies]

Use only information from the records. If something is missing, state "Not documented".

DISCHARGE SUMMARY:""",
            input_variables=["context"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"context": context[:6000]})
        
        return summary

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("\n📁 Step 1: Creating Sample Documents")
    print("=" * 60)
    
    doc_dir = create_sample_documents()
    file_path = str(doc_dir / "patient_complete_record.txt")
    
    print("\n" + "=" * 60)
    print("🤖 Step 2: Initializing RAG System")
    print("=" * 60)
    
    rag = LocalRAG()
    rag.ingest(file_path)
    
    print("\n" + "=" * 60)
    print("🔍 Step 3: Testing Queries")
    print("=" * 60)
    
    questions = [
        "What was the patient's admission diagnosis?",
        "What medications are prescribed at discharge?",
        "What are the patient's allergies?",
        "What was the patient's temperature on admission?",
        "What is the follow-up plan?"
    ]
    
    for q in questions:
        print(f"\n❓ {q}")
        try:
            answer = rag.query(q)
            print(f"💡 {answer}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📄 Step 4: Generating Complete Discharge Summary")
    print("=" * 60)
    
    try:
        summary = rag.generate_discharge_summary()
        print("\n" + "=" * 60)
        print("FINAL DISCHARGE SUMMARY")
        print("=" * 60)
        print(summary)
        
        # Save to file
        output_file = "discharge_summary_local.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n✅ Summary saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️ Error generating summary: {e}")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
