"""
Batch Pipeline Comparison Script
Run multiple test questions across all pipelines and generate comparison report
"""
import time
import json
from datetime import datetime
from pipeline_1_llm import get_pipeline_1_metadata
from pipeline_2_rag import get_pipeline_2_metadata
from pipeline_3_graphrag import get_pipeline_3_metadata


# Test questions for credit card fraud dataset
TEST_QUESTIONS = [
    "What are the core metrics tracked in the dataset?",
    "How can we identify fraudulent transactions?",
    "What patterns exist in the transaction data?",
    "Explain the key features in the dataset",
    "What time periods show the highest activity?",
]


def run_pipeline_comparison(question, verbose=True):
    """
    Run a question through all three pipelines and collect results
    
    Args:
        question: Question to test
        verbose: Print progress
    
    Returns:
        dict: Results from all pipelines with timing
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
    
    results = {}
    
    # Pipeline 1
    if verbose:
        print("\n🔴 Running Pipeline 1 (LLM-Only)...")
    start = time.time()
    try:
        result_1 = get_pipeline_1_metadata(question)
        time_1 = time.time() - start
        results['pipeline_1'] = {
            'result': result_1,
            'time': time_1,
            'success': True
        }
        if verbose:
            print(f"   ✓ Completed in {time_1:.2f}s")
    except Exception as e:
        results['pipeline_1'] = {
            'error': str(e),
            'time': 0,
            'success': False
        }
        if verbose:
            print(f"   ✗ Error: {e}")
    
    # Pipeline 2
    if verbose:
        print("\n🔵 Running Pipeline 2 (Vector RAG)...")
    start = time.time()
    try:
        result_2 = get_pipeline_2_metadata(question)
        time_2 = time.time() - start
        results['pipeline_2'] = {
            'result': result_2,
            'time': time_2,
            'success': True
        }
        if verbose:
            print(f"   ✓ Completed in {time_2:.2f}s")
    except Exception as e:
        results['pipeline_2'] = {
            'error': str(e),
            'time': 0,
            'success': False
        }
        if verbose:
            print(f"   ✗ Error: {e}")
    
    # Pipeline 3
    if verbose:
        print("\n🟢 Running Pipeline 3 (GraphRAG)...")
    start = time.time()
    try:
        result_3 = get_pipeline_3_metadata(question)
        time_3 = time.time() - start
        results['pipeline_3'] = {
            'result': result_3,
            'time': time_3,
            'success': True
        }
        if verbose:
            print(f"   ✓ Completed in {time_3:.2f}s")
    except Exception as e:
        results['pipeline_3'] = {
            'error': str(e),
            'time': 0,
            'success': False
        }
        if verbose:
            print(f"   ✗ Error: {e}")
    
    return results


def print_comparison_table(all_results):
    """Print a formatted comparison table"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    # Calculate averages
    avg_times = {}
    success_rates = {}
    
    for pipeline in ['pipeline_1', 'pipeline_2', 'pipeline_3']:
        times = [r[pipeline]['time'] for r in all_results if r[pipeline]['success']]
        successes = sum(1 for r in all_results if r[pipeline]['success'])
        
        avg_times[pipeline] = sum(times) / len(times) if times else 0
        success_rates[pipeline] = (successes / len(all_results)) * 100
    
    # Print table
    print(f"\n{'Pipeline':<25} {'Avg Time':<15} {'Success Rate':<15}")
    print("-" * 55)
    
    pipelines = [
        ('pipeline_1', 'Pipeline 1: LLM-Only'),
        ('pipeline_2', 'Pipeline 2: Vector RAG'),
        ('pipeline_3', 'Pipeline 3: GraphRAG'),
    ]
    
    for key, name in pipelines:
        avg_time = f"{avg_times[key]:.2f}s"
        success = f"{success_rates[key]:.0f}%"
        print(f"{name:<25} {avg_time:<15} {success:<15}")
    
    print("\n" + "="*80)


def save_results(all_results, filename=None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = []
    for result in all_results:
        serializable = {}
        for pipeline, data in result.items():
            if data['success']:
                # Remove non-serializable objects
                clean_result = {
                    'pipeline': data['result']['pipeline'],
                    'answer': data['result']['answer'],
                    'retrieval_method': data['result']['retrieval_method'],
                    'time': data['time']
                }
                serializable[pipeline] = clean_result
            else:
                serializable[pipeline] = {
                    'error': data['error'],
                    'time': data['time']
                }
        serializable_results.append(serializable)
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'questions': TEST_QUESTIONS,
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {filename}")


def main():
    """Run batch comparison"""
    print("="*80)
    print("RAG PIPELINE BATCH COMPARISON")
    print("="*80)
    print(f"\nTesting {len(TEST_QUESTIONS)} questions across all pipelines...")
    
    all_results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}]", end=" ")
        results = run_pipeline_comparison(question, verbose=True)
        all_results.append(results)
        
        # Brief summary
        print("\n📊 Quick Summary:")
        for pipeline, data in results.items():
            if data['success']:
                answer_preview = data['result']['answer'][:100] + "..."
                print(f"  {pipeline}: {data['time']:.2f}s")
                print(f"    → {answer_preview}")
    
    # Overall summary
    print_comparison_table(all_results)
    
    # Save results
    save_results(all_results)
    
    print("\n✅ Batch comparison complete!")
    print("\n💡 Tips:")
    print("  • Use these results to demonstrate pipeline differences")
    print("  • Compare answer quality across pipelines")
    print("  • Highlight performance trade-offs")
    print("  • Show how context improves accuracy\n")


if __name__ == "__main__":
    main()
