"""
Pipeline 1: LLM-Only (Baseline)
No retrieval, just direct LLM inference
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config


def get_pipeline_1_answer(question):
    """
    Pipeline 1: Direct LLM inference without any retrieval
    This is the baseline/worst-case scenario
    
    Args:
        question: User's question
    
    Returns:
        str: LLM's answer based solely on its training data
    """
    try:
        # Validate and set API key
        Config.validate()
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, 
            temperature=Config.LLM_TEMPERATURE
        )
        
        # Simple prompt - LLM uses only its own knowledge
        template = """You are a helpful assistant. Answer the following question based on your knowledge.
        
Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Build chain
        chain = prompt | llm | StrOutputParser()
        
        # Get answer
        answer = chain.invoke({"question": question})
        
        return answer
        
    except Exception as e:
        return f"Error in Pipeline 1: {str(e)}"


def get_pipeline_1_metadata(question):
    """
    Get answer with metadata for comparison
    
    Returns:
        dict: Answer and metadata
    """
    answer = get_pipeline_1_answer(question)
    
    return {
        "answer": answer,
        "sources": [],  # No sources in LLM-only
        "pipeline": "Pipeline 1: LLM-Only",
        "retrieval_method": "None",
        "context_used": False
    }


if __name__ == "__main__":
    print("Testing Pipeline 1 (LLM-Only)...")
    print("=" * 50)
    
    test_question = "What are the core metrics tracked in the dataset?"
    print(f"Question: {test_question}\n")
    
    result = get_pipeline_1_metadata(test_question)
    print(f"Pipeline: {result['pipeline']}")
    print(f"Answer: {result['answer']}")
    print(f"\nNote: This answer is based solely on the LLM's training data,")
    print("not on any specific dataset you provided.")
