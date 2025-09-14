#test_classifier.py
from src.model_inference import QueryClassifier
import json

def test_classification():
    print("=== TESTING QUERY CLASSIFIER ===")
    
    # Initialize classifier
    classifier = QueryClassifier()
    
    if not classifier.model:
        print("❌ Model not loaded properly")
        return False
    
    # Print device info
    print(f"Using device: {classifier.device}")
    if hasattr(classifier, 'get_gpu_memory_info'):
        classifier.get_gpu_memory_info()
    
    # Test queries - start with simplest
    test_queries = [
        "a student interested in the Student Mentorship Program for the 2025-26 academic year. I would like to know if there are any specific guidelines or eligibility criteria mentioned in the brochure that I should be aware of before applying.",
        "Hello",  # Very simple test
        "When is the exam?",  # Simple question
        "I forgot my password to the student portal",
        "Can you help me with my assignment submission?",
        "How do I register for next semester courses?",
        "What are the fees for the upcoming semester?"
    ]
    
    results = []
    successful_tests = 0
    
    print("\n🧪 Testing with sample queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")
        
        # Clear cache before each test
        if hasattr(classifier, 'clear_gpu_cache'):
            classifier.clear_gpu_cache()
        
        try:
            result = classifier.classify_and_respond(query)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                # Continue with other tests instead of breaking
                continue
            
            print(f"✅ Category: {result['category']}")
            print(f"✅ Action: {result['action']}")
            print(f"✅ Confidence: {result['confidence']:.2f}")
            print(f"✅ Response: {result['response'][:100]}...")
            
            results.append({
                "query": query,
                "result": result
            })
            successful_tests += 1
            
        except Exception as e:
            print(f"❌ Test {i} failed with exception: {e}")
            continue
        
        # Memory info after each test
        if hasattr(classifier, 'get_gpu_memory_info'):
            classifier.get_gpu_memory_info()
    
    # Save test results
    import os
    os.makedirs('models', exist_ok=True)
    with open('models/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Testing completed!")
    print(f"✅ Successful tests: {successful_tests}/{len(test_queries)}")
    print(f"📁 Results saved to models/test_results.json")
    
    return successful_tests > 0

if __name__ == "__main__":
    test_classification()