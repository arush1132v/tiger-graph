"""
Pipeline 3: GraphRAG (Graph + LLM)
Uses TigerGraph for entity extraction, relationship mapping, and multi-hop reasoning
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import Config


def get_pipeline_3_answer(question):
    """
    Pipeline 3: GraphRAG - Knowledge graph-based retrieval
    
    This is a placeholder implementation. Replace with actual TigerGraph integration:
    1. Extract entities from question
    2. Query TigerGraph for relevant subgraph
    3. Perform multi-hop reasoning
    4. Use structured context for LLM generation
    
    Args:
        question: User's question
    
    Returns:
        str: Answer based on graph-structured context
    """
    try:
        # Validate and set API key
        Config.validate()
        os.environ["GOOGLE_API_KEY"] = Config.GOOGLE_API_KEY
        
        # TODO: Replace with TigerGraph GraphRAG implementation
        # Example workflow:
        # 1. entities = extract_entities(question)
        # 2. subgraph = tigergraph_query(entities)
        # 3. context = format_graph_context(subgraph)
        
        # Placeholder: Using LLM with simulated graph context
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, 
            temperature=Config.LLM_TEMPERATURE
        )
        
        template = """You are answering based on a knowledge graph structure.

[PLACEHOLDER: Graph context would go here]
- Entities: {entities}
- Relationships: {relationships}
- Multi-hop paths: {paths}

Question: {question}

Answer based on the graph structure above:"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        # Placeholder graph data
        answer = chain.invoke({
            "question": question,
            "entities": "[To be extracted from TigerGraph]",
            "relationships": "[To be queried from TigerGraph]",
            "paths": "[Multi-hop reasoning from TigerGraph]"
        })
        
        return f"[PLACEHOLDER - Integrate TigerGraph here]\n\n{answer}"
        
    except Exception as e:
        return f"Error in Pipeline 3: {str(e)}"


def get_pipeline_3_metadata(question):
    """
    Get answer with metadata for comparison
    
    Returns:
        dict: Answer and metadata
    """
    answer = get_pipeline_3_answer(question)
    
    return {
        "answer": answer,
        "sources": [],  # Will be populated with graph nodes/edges
        "pipeline": "Pipeline 3: GraphRAG",
        "retrieval_method": "Knowledge Graph (TigerGraph)",
        "context_used": True,
        "graph_hops": 0  # Placeholder
    }


# Integration guide for TigerGraph
TIGERGRAPH_INTEGRATION_GUIDE = """
TigerGraph GraphRAG Integration Steps:
======================================

1. Install TigerGraph Dependencies:
   pip install pyTigerGraph

2. Setup TigerGraph Connection:
   import pyTigerGraph as tg
   conn = tg.TigerGraphConnection(
       host="YOUR_HOST",
       graphname="YOUR_GRAPH",
       username="YOUR_USERNAME",
       password="YOUR_PASSWORD"
   )

3. Key Functions to Implement:
   
   a) extract_entities(question):
      - Use NER or LLM to extract entities from question
      - Return list of entity names/types
   
   b) query_graph(entities, max_hops=2):
      - Query TigerGraph for subgraph around entities
      - Perform multi-hop traversal
      - Return nodes and edges
   
   c) format_graph_context(subgraph):
      - Convert graph structure to text context
      - Include entity properties and relationships
      - Format for LLM consumption
   
   d) generate_answer(question, graph_context):
      - Use LLM with graph-structured context
      - Generate final answer

4. Example TigerGraph Query:
   results = conn.runInstalledQuery(
       "entity_neighborhood",
       params={"entity": entity_name, "hops": 2}
   )

5. Replace the placeholder code above with actual TigerGraph calls
"""


if __name__ == "__main__":
    print("Pipeline 3 (GraphRAG) - Placeholder")
    print("=" * 50)
    print(TIGERGRAPH_INTEGRATION_GUIDE)
    print("\n" + "=" * 50)
    
    test_question = "What are the core metrics tracked in the dataset?"
    print(f"\nTesting with question: {test_question}\n")
    
    result = get_pipeline_3_metadata(test_question)
    print(f"Pipeline: {result['pipeline']}")
    print(f"Answer:\n{result['answer']}")
