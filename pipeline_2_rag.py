"""
Pipeline 2: Basic RAG (Vector + LLM)
Vector embeddings retrieve similar chunks, LLM generates answer using context
"""
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import Config


def get_pipeline_2_answer(question, return_sources=False):
    """
    Pipeline 2: Vector RAG - Retrieve relevant chunks and generate answer
    
    Args:
        question: User's question
        return_sources: If True, return (answer, sources) tuple
    
    Returns:
        str or tuple: Answer, optionally with source documents
    """
    try:
        # Validate and set API key
        Config.validate()
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        
        # Load existing vector database
        vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR, 
            embedding_function=GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        )
        
        # Setup retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        # Setup LLM
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, 
            temperature=Config.LLM_TEMPERATURE
        )
        
        # Improved prompt template
        template = """You are a helpful assistant that answers questions based on the provided context.

Context Information:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information, say so
- Be specific and cite relevant details from the context
- Keep your answer clear and concise

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(retrieved_docs):
            """Format retrieved documents with metadata"""
            formatted = []
            for i, doc in enumerate(retrieved_docs, 1):
                content = doc.page_content
                # Add source metadata if available
                source = doc.metadata.get('source', 'Unknown')
                formatted.append(f"[Source {i} - {source}]\n{content}")
            return "\n\n".join(formatted)
        
        # Build RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Get answer
        answer = rag_chain.invoke(question)
        
        if return_sources:
            # Retrieve source documents for reference
            source_docs = retriever.invoke(question)
            return answer, source_docs
        
        return answer
        
    except Exception as e:
        error_msg = f"Error in Pipeline 2: {str(e)}"
        if return_sources:
            return error_msg, []
        return error_msg


def get_pipeline_2_metadata(question):
    """
    Get answer with full metadata for comparison
    
    Returns:
        dict: Answer, sources, and metadata
    """
    answer, sources = get_pipeline_2_answer(question, return_sources=True)
    
    # Extract source information
    source_list = []
    for i, doc in enumerate(sources, 1):
        source_list.append({
            "id": i,
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        })
    
    return {
        "answer": answer,
        "sources": source_list,
        "pipeline": "Pipeline 2: Vector RAG",
        "retrieval_method": "Vector Similarity (Embeddings)",
        "context_used": True,
        "num_sources": len(sources)
    }


if __name__ == "__main__":
    print("Testing Pipeline 2 (Vector RAG)...")
    print("=" * 50)
    
    test_question = "What are the core metrics tracked in the dataset?"
    print(f"Question: {test_question}\n")
    
    result = get_pipeline_2_metadata(test_question)
    print(f"Pipeline: {result['pipeline']}")
    print(f"Retrieval Method: {result['retrieval_method']}")
    print(f"Sources Retrieved: {result['num_sources']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources Used:")
    for source in result['sources']:
        print(f"\n  Source {source['id']}:")
        print(f"  {source['content']}")
