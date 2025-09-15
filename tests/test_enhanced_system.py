from src.enhanced_decision_engine import EnhancedDecisionEngine
import json

# test_enhanced_system.py (continued)
def test_enhanced_system():
    print("=== TESTING ENHANCED SYSTEM WITH RAG ===")
    
    # Initialize enhanced system
    print("Initializing enhanced decision engine...")
    engine = EnhancedDecisionEngine()
    
    # Get system stats
    stats = engine.get_system_stats()
    print(f"System Stats:")
    print(f"Model loaded: {stats['model_loaded']}")
    print(f"RAG enabled: {stats['rag_enabled']}")
    if 'knowledge_base' in stats:
        print(f"Knowledge base: {stats['knowledge_base']['total_chunks']} chunks")

    if not stats['model_loaded']:
        print("Model not loaded - cannot proceed")
        return False
    
    # Enhanced test queries with expected RAG context
    enhanced_test_queries = [
        {
            'id': 'rag_test_1',
            'subject': 'Final Exam Date',
            'body': 'Can you tell me when the final examinations are scheduled for this semester?',
            'sender': 'student@university.edu',
            'expected_rag': True,
            'expected_category': 'academic'
        },
        {
            'id': 'rag_test_2',
            'subject': 'Fee Payment Methods',
            'body': 'What are the available payment methods for semester fees? Can I pay in installments?',
            'sender': 'student@university.edu',
            'expected_rag': True,
            'expected_category': 'administrative'
        },
        {
            'id': 'rag_test_3',
            'subject': 'Password Reset Help',
            'body': 'I cannot access my student portal account. I think I forgot my password. How can I reset it?',
            'sender': 'student@university.edu',
            'expected_rag': True,
            'expected_category': 'technical'
        },
        {
            'id': 'rag_test_4',
            'subject': 'Assignment Late Submission',
            'body': 'I missed the assignment deadline due to illness. What is the penalty and can I still submit?',
            'sender': 'student@university.edu',
            'expected_rag': True,
            'expected_category': 'academic'
        },
        {
            'id': 'rag_test_5',
            'subject': 'Urgent Issue',
            'body': 'This is very urgent! I am having serious problems with my registration and nobody is helping me!',
            'sender': 'student@university.edu',
            'expected_rag': False,  # Should escalate due to urgent/problem keywords
            'expected_action': 'escalate'
        }
    ]
    
    print(f"\nTesting with {len(enhanced_test_queries)} enhanced queries...")
    
    results = []
    action_counts = {'auto_respond': 0, 'review_needed': 0, 'escalate': 0}
    rag_usage_count = 0
    
    for i, test_email in enumerate(enhanced_test_queries, 1):
        print(f"\n--- Enhanced Test {i}: {test_email['id']} ---")
        print(f"Query: {test_email['subject']}")
        
        try:
            result = engine.process_query(test_email)
            
            # Count actions
            action = result['action']
            action_counts[action] += 1
            
            # Check RAG usage
            rag_sources = result.get('rag_sources', [])
            if rag_sources:
                rag_usage_count += 1
            
            print(f"Action: {action}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Category: {result.get('category', 'N/A')}")
            print(f"RAG Sources: {len(rag_sources)} sources")

            if rag_sources:
                print(f"   Sources: {', '.join(rag_sources[:2])}")
            
            if result.get('response'):
                response_preview = result['response'][:150] + "..." if len(result['response']) > 150 else result['response']
                print(f"   Response: {response_preview}")
            
            # Validation
            validation_passed = True
            
            # Check expected action
            if 'expected_action' in test_email:
                if result['action'] != test_email['expected_action']:
                    print(f"Expected action: {test_email['expected_action']}, got: {result['action']}")
                    validation_passed = False
            
            # Check RAG usage expectation
            if test_email.get('expected_rag', True) and not rag_sources:
                print("Expected RAG context but none found")
                validation_passed = False
            
            if validation_passed:
                print("Validation passed")
            
            results.append({
                'test_id': test_email['id'],
                'query': test_email['subject'],
                'result': result,
                'validation_passed': validation_passed
            })
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append({
                'test_id': test_email['id'],
                'query': test_email['subject'],
                'result': {'error': str(e)},
                'validation_passed': False
            })
    
    # Summary
    print(f"\nENHANCED SYSTEM TEST SUMMARY")
    print(f"Total tests: {len(enhanced_test_queries)}")
    print(f"Action distribution:")
    for action, count in action_counts.items():
        percentage = (count / len(enhanced_test_queries)) * 100
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    print(f"RAG usage: {rag_usage_count}/{len(enhanced_test_queries)} queries ({rag_usage_count/len(enhanced_test_queries)*100:.1f}%)")
    
    passed_tests = sum(1 for r in results if r['validation_passed'])
    print(f"Validation passed: {passed_tests}/{len(enhanced_test_queries)} ({passed_tests/len(enhanced_test_queries)*100:.1f}%)")
    
    # Save detailed results
    with open('models/enhanced_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to: models/enhanced_test_results.json")
    
    # Quality checks
    quality_passed = True
    
    if stats['rag_enabled'] and rag_usage_count == 0:
        print("RAG enabled but not used in any queries")
        quality_passed = False
    
    if action_counts['escalate'] > len(enhanced_test_queries) * 0.8:
        print("Too many queries escalated - system may be too conservative")
        quality_passed = False
    
    if action_counts['auto_respond'] == 0:
        print("No queries auto-responded - system may not be working properly")
        quality_passed = False
    
    if quality_passed:
        print("Quality checks passed")
        return True
    else:
        print("Some quality checks failed")
        return False

if __name__ == "__main__":
    test_enhanced_system()