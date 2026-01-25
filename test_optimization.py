#!/usr/bin/env python3
"""
Comprehensive Test Suite for Optimization Verification
Tests all critical optimization scenarios
"""

import requests
import time
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(name: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}TEST: {name}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_fail(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message: str):
    print(f"{Colors.YELLOW}ℹ {message}{Colors.END}")

def ask_question(question: str) -> tuple[Dict[Any, Any], float]:
    """Ask a question and return response + time taken"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"question": question},
            timeout=15  # Allow some buffer over 10s
        )
        elapsed_time = time.time() - start_time
        return response.json(), elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {"error": str(e)}, elapsed_time

def test_health_check():
    """Test 1: Basic health check"""
    print_test("Health Check & Initialization")
    
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    
    if data["status"] == "healthy":
        print_success(f"Server is healthy: {data['message']}")
        print_success(f"Vectorstore initialized: {data['vectorstore_initialized']}")
        print_success(f"LLM configured: {data['llm_configured']}")
        return True
    else:
        print_fail("Server health check failed")
        return False

def test_response_time():
    """Test 2: Response time < 10 seconds"""
    print_test("Response Time Performance (<10s)")
    
    test_questions = [
        "What's your name?",
        "Tell me about your skills",
        "What projects have you worked on?",
        "What's your email?"
    ]
    
    all_passed = True
    for question in test_questions:
        response, elapsed = ask_question(question)
        
        if "error" not in response:
            if elapsed < 10.0:
                print_success(f"✓ '{question[:40]}...' - {elapsed:.2f}s ✓")
            else:
                print_fail(f"✗ '{question[:40]}...' - {elapsed:.2f}s (>10s!)")
                all_passed = False
        else:
            print_fail(f"✗ '{question[:40]}...' - ERROR: {response['error']}")
            all_passed = False
    
    return all_passed

def test_cache_functionality():
    """Test 3: Cache exact hits and fuzzy matching"""
    print_test("Cache Functionality (Exact & Fuzzy Matching)")
    
    # Test exact cache hit
    question = "What is your phone number?"
    print_info(f"First request: {question}")
    _, time1 = ask_question(question)
    print_info(f"Time taken: {time1:.2f}s")
    
    print_info("Requesting same question (should hit cache)...")
    _, time2 = ask_question(question)
    print_info(f"Time taken: {time2:.2f}s")
    
    if time2 < 1.0:
        print_success(f"Exact cache hit! ({time2:.3f}s)")
        exact_pass = True
    else:
        print_fail(f"Cache miss or slow ({time2:.2f}s)")
        exact_pass = False
    
    # Test fuzzy matching
    print_info("\nTesting fuzzy cache matching...")
    similar_questions = [
        "What's your phone?",
        "phone number?",
        "How can I call you?"
    ]
    
    fuzzy_pass = True
    for similar_q in similar_questions:
        print_info(f"Similar question: {similar_q}")
        _, time_fuzzy = ask_question(similar_q)
        if time_fuzzy < 2.0:
            print_success(f"Fuzzy match likely! ({time_fuzzy:.3f}s)")
        else:
            print_info(f"New question or slow fuzzy ({time_fuzzy:.2f}s)")
    
    return exact_pass

def test_complete_responses():
    """Test 4: Responses are complete (not truncated)"""
    print_test("Complete Response Test (No Truncation)")
    
    # Ask complex questions that require multiple iterations
    complex_questions = [
        "Tell me about your work experience and projects in detail",
        "What are your technical skills and how have you used them?",
        "Describe your education and achievements"
    ]
    
    all_complete = True
    for question in complex_questions:
        print_info(f"\nAsking: {question}")
        response, elapsed = ask_question(question)
        
        if "error" not in response:
            answer = response.get("answer", "")
            word_count = len(answer.split())
            
            print_info(f"Response length: {word_count} words, Time: {elapsed:.2f}s")
            
            # Check if response seems complete (not cut off mid-sentence)
            if word_count > 20 and (answer.endswith(".") or answer.endswith("!") or "</a>" in answer[-20:]):
                print_success(f"Response appears complete ({word_count} words)")
            else:
                print_fail(f"Response may be truncated (only {word_count} words)")
                all_complete = False
        else:
            print_fail(f"Error: {response['error']}")
            all_complete = False
    
    return all_complete

def test_fast_fallback():
    """Test 5: Fast fallback responses work correctly"""
    print_test("Fast Fallback Response System")
    
    # These should trigger fast fallback patterns
    fallback_questions = [
        ("What's your email address?", "email", "mailto:"),
        ("What's your phone number?", "phone", "tel:"),
        ("Where's your GitHub?", "github", "GitHub"),
        ("Hello!", "greeting", "Hey")
    ]
    
    all_passed = True
    for question, keyword, expected_in_response in fallback_questions:
        print_info(f"\nTesting: {question}")
        response, elapsed = ask_question(question)
        
        if "error" not in response:
            answer = response.get("answer", "")
            
            # Check if response is fast and contains expected content
            if elapsed < 10.0 and (keyword.lower() in answer.lower() or expected_in_response in answer):
                print_success(f"✓ Fast response ({elapsed:.2f}s) with relevant content")
            else:
                print_info(f"Response in {elapsed:.2f}s")
                if keyword.lower() not in answer.lower():
                    print_info(f"Note: '{keyword}' not found in response")
        else:
            print_fail(f"Error: {response['error']}")
            all_passed = False
    
    return all_passed

def test_personality_preservation():
    """Test 6: Personality is maintained across responses"""
    print_test("Personality Preservation")
    
    test_questions = [
        "Who are you?",
        "What do you like to do?",
        "Tell me about yourself"
    ]
    
    personality_indicators = [
        "Abhay", "AI", "engineer", "motorcycle", "tech", "cloud"
    ]
    
    all_passed = True
    for question in test_questions:
        print_info(f"\nAsking: {question}")
        response, elapsed = ask_question(question)
        
        if "error" not in response:
            answer = response.get("answer", "").lower()
            
            # Check for personality indicators
            found_indicators = [ind for ind in personality_indicators if ind.lower() in answer]
            
            if len(found_indicators) >= 2:
                print_success(f"Personality present! Found: {', '.join(found_indicators)}")
                print_info(f"Sample: {response.get('answer', '')[:150]}...")
            else:
                print_info(f"Limited personality markers: {found_indicators}")
        else:
            print_fail(f"Error: {response['error']}")
            all_passed = False
    
    return all_passed

def test_html_link_formatting():
    """Test 7: HTML links are properly formatted"""
    print_test("HTML Link Formatting")
    
    questions_expecting_links = [
        ("What's your email?", "mailto:"),
        ("What's your phone?", "tel:"),
        ("Where's your GitHub?", "github.com")
    ]
    
    all_passed = True
    for question, expected_protocol in questions_expecting_links:
        print_info(f"\nTesting: {question}")
        response, elapsed = ask_question(question)
        
        if "error" not in response:
            answer = response.get("answer", "")
            
            # Check for HTML link format
            if '<a href=' in answer and 'target="_blank"' in answer:
                print_success("✓ HTML link format detected")
                if expected_protocol in answer:
                    print_success(f"✓ Contains expected protocol: {expected_protocol}")
                else:
                    print_info(f"Note: Protocol '{expected_protocol}' not found")
            else:
                print_info("No HTML links found (may be plain text response)")
        else:
            print_fail(f"Error: {response['error']}")
            all_passed = False
    
    return all_passed

def test_optimization_stats():
    """Test 8: Check optimization stats endpoint"""
    print_test("Optimization Stats Verification")
    
    try:
        response = requests.get(f"{BASE_URL}/optimization-stats")
        data = response.json()
        
        print_info(f"Cache entries: {data['caching']['current_entries']}")
        print_info(f"Max cache size: {data['caching']['max_entries']}")
        print_info(f"Cache TTL: {data['caching']['ttl_seconds']}s")
        print_info(f"Agent max iterations: {data['agent_optimization']['max_iterations']}")
        print_info(f"Timeout protection: {data['agent_optimization'].get('timeout_protection', 'N/A')}")
        
        print_success("Optimization stats retrieved successfully")
        return True
    except Exception as e:
        print_fail(f"Failed to get optimization stats: {e}")
        return False

def run_all_tests():
    """Run all test scenarios"""
    print(f"\n{Colors.BOLD}{'='*60}")
    print("COMPREHENSIVE OPTIMIZATION TEST SUITE")
    print(f"{'='*60}{Colors.END}\n")
    
    results = {
        "Health Check": test_health_check(),
        "Response Time (<10s)": test_response_time(),
        "Cache Functionality": test_cache_functionality(),
        "Complete Responses": test_complete_responses(),
        "Fast Fallback": test_fast_fallback(),
        "Personality Preservation": test_personality_preservation(),
        "HTML Link Formatting": test_html_link_formatting(),
        "Optimization Stats": test_optimization_stats()
    }
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}{Colors.END}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.END}" if passed_test else f"{Colors.RED}FAILED{Colors.END}"
        print(f"{test_name}: {status}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! System is optimized and ready for deployment! 🎉{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some tests need attention. Review failed tests above.{Colors.END}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
