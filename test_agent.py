#!/usr/bin/env python3
"""
Test script for Abhay's AI Agent
Run this to test your FastAPI backend locally with various question types
"""

import requests
import json
import time
import sys
from typing import Dict, Any
from colorama import Fore, Style, init


# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Configuration
BASE_URL = "http://0.0.0.0:8000"
TIMEOUT = 30
class AgentTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def print_header(self, text: str):
        """Print a colored header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text.center(60)}")
        print(f"{Fore.CYAN}{'='*60}")
        
    def print_success(self, text: str):
        """Print success message"""
        print(f"{Fore.GREEN}✓ {text}")
        
    def print_error(self, text: str):
        """Print error message"""
        print(f"{Fore.RED}✗ {text}")
        
    def print_info(self, text: str):
        """Print info message"""
        print(f"{Fore.YELLOW}ℹ {text}")
        
    def print_response(self, response: Dict[Any, Any], show_details: bool = True):
        """Print formatted API response"""
        answer = response.get('answer', 'No answer provided')
        confidence = response.get('confidence', 0)
        used_resume = response.get('used_resume', False)
        sources = response.get('sources', [])
        
        print(f"\n{Fore.MAGENTA}AI Response:")
        print(f"{Style.BRIGHT}{answer}")
        
        if show_details:
            print(f"\n{Fore.CYAN}Details:")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Used Resume: {'Yes' if used_resume else 'No'}")
            if sources:
                print(f"  Sources: {len(sources)} found")
                for i, source in enumerate(sources[:2], 1):
                    print(f"    {i}. {source[:50]}...")
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Health check passed - Status: {data.get('status')}")
                self.print_info(f"Vectorstore initialized: {data.get('vectorstore_initialized')}")
                self.print_info(f"LLM configured: {data.get('llm_configured')}")
                return True
            else:
                self.print_error(f"Health check failed - Status: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Health check failed - Error: {str(e)}")
            return False
    
    def ask_question(self, question: str, show_details: bool = True) -> Dict[Any, Any]:
        """Send a question to the agent"""
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/ask", 
                json=payload, 
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success("Question processed successfully")
                self.print_response(data, show_details)
                return data
            else:
                self.print_error(f"Question failed - Status: {response.status_code}")
                try:
                    error_data = response.json()
                    self.print_error(f"Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    self.print_error(f"Response: {response.text}")
                return {}
                
        except Exception as e:
            self.print_error(f"Question failed - Error: {str(e)}")
            return {}
    
    def test_resume_detection(self, question: str):
        """Test if a question triggers resume detection"""
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/test-resume-detection", 
                json=payload, 
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                will_use = data.get('will_use_resume', False)
                matched_keywords = data.get('matched_keywords', [])
                
                print(f"\n{Fore.CYAN}Resume Detection Test:")
                print(f"  Question: {question}")
                print(f"  Will use resume: {'Yes' if will_use else 'No'}")
                print(f"  Matched keywords: {matched_keywords}")
                print(f"  Recommendation: {data.get('recommendation', 'N/A')}")
                return data
            else:
                self.print_error(f"Detection test failed - Status: {response.status_code}")
                return {}
                
        except Exception as e:
            self.print_error(f"Detection test failed - Error: {str(e)}")
            return {}
    
    def get_info(self):
        """Get API information"""
        try:
            response = self.session.get(f"{self.base_url}/info", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                print(f"\n{Fore.CYAN}API Information:")
                print(f"  Version: {data.get('api_version')}")
                print(f"  Owner: {data.get('owner')}")
                print(f"  Description: {data.get('description')}")
                print(f"  Capabilities: {len(data.get('capabilities', []))} features")
                return data
            else:
                self.print_error(f"Info request failed - Status: {response.status_code}")
                return {}
        except Exception as e:
            self.print_error(f"Info request failed - Error: {str(e)}")
            return {}
    
    def update_training_prompt(self, prompt: str):
        """Update the training prompt"""
        try:
            payload = {"training_prompt": prompt}
            response = self.session.post(
                f"{self.base_url}/training-prompt", 
                json=payload, 
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success("Training prompt updated successfully")
                print(f"  Status: {data.get('status')}")
                print(f"  Message: {data.get('message')}")
                return data
            else:
                self.print_error(f"Training update failed - Status: {response.status_code}")
                return {}
                
        except Exception as e:
            self.print_error(f"Training update failed - Error: {str(e)}")
            return {}

def run_comprehensive_test():
    """Run a comprehensive test suite"""
    tester = AgentTester()
    
    # Test questions
    test_questions = [
        # General conversation (should NOT use resume)
        {
            "category": "General Conversation", 
            "questions": [
                "Tell me about Abhay's Citizenship",
                "Tell me about Abhay's hobbies",
                "What's 15 * 23?",
                "Tell me about the weather today",
                "What do you think about the latest tech trends?"
            ]
        },
        # Resume-related (SHOULD use resume)
        {
            "category": "Professional Background", 
            "questions": [
                "Tell me about your work experience",
                "What's your educational background?",
                "What programming languages do you know?",
                "Tell me about your projects",
                "What achievements do you have?",
                "Where did you do your internship?",
                "What's your GPA?"
            ]
        },
        # Edge cases
        {
            "category": "Edge Cases", 
            "questions": [
                "Calculate my work experience in months if I started in January 2024",
                "What skills would be good for AI development?",
                "How would you solve a complex programming problem?"
            ]
        }
    ]
    
    print(f"{Fore.GREEN}{Style.BRIGHT}🚀 Starting Comprehensive Test Suite for Abhay's AI Agent")
    print(f"{Fore.BLUE}Testing URL: {tester.base_url}")
    
    # Health check
    tester.print_header("HEALTH CHECK")
    if not tester.test_health():
        print(f"{Fore.RED}❌ Health check failed. Make sure your FastAPI server is running!")
        return False
    
    # API Info
    tester.print_header("API INFORMATION")
    tester.get_info()
    
    # Test all question categories
    for category_data in test_questions:
        category = category_data["category"]
        questions = category_data["questions"]
        
        tester.print_header(f"TESTING: {category.upper()}")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{Fore.YELLOW}Test {i}/{len(questions)}: {question}")
            
            # Test resume detection first
            tester.test_resume_detection(question)
            
            # Ask the actual question
            print(f"{Fore.BLUE}Asking agent...")
            result = tester.ask_question(question, show_details=True)
            
            # Wait a bit between requests
            if i < len(questions):
                time.sleep(1)
    
    tester.print_header("TEST SUMMARY")
    tester.print_success("Comprehensive test completed!")
    print(f"{Fore.CYAN}💡 Check the responses above to see if:")
    print(f"   • General questions avoided using resume data")
    print(f"   • Professional questions used resume data appropriately")
    print(f"   • The AI maintained Abhay's personality throughout")
    
    return True

def interactive_mode():
    """Run interactive testing mode"""
    tester = AgentTester()
    
    print(f"{Fore.GREEN}{Style.BRIGHT}🎯 Interactive Test Mode for Abhay's AI Agent")
    print(f"{Fore.BLUE}Testing URL: {tester.base_url}")
    print(f"{Fore.YELLOW}Type 'quit', 'exit', or 'q' to exit")
    print(f"{Fore.YELLOW}Type 'health' to check system health")
    print(f"{Fore.YELLOW}Type 'info' to get API information")
    print(f"{Fore.YELLOW}Type 'detect <question>' to test resume detection")
    
    # Initial health check
    if not tester.test_health():
        print(f"{Fore.RED}❌ Health check failed. Make sure your FastAPI server is running!")
        return
    
    while True:
        try:
            print(f"\n{Fore.CYAN}{'─'*50}")
            question = input(f"{Fore.WHITE}Ask Abhay: {Style.RESET_ALL}").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print(f"{Fore.GREEN}👋 Goodbye!")
                break
            elif question.lower() == 'health':
                tester.test_health()
            elif question.lower() == 'info':
                tester.get_info()
            elif question.lower().startswith('detect '):
                test_q = question[7:]  # Remove 'detect '
                if test_q:
                    tester.test_resume_detection(test_q)
                else:
                    tester.print_error("Please provide a question to test")
            else:
                # Ask the question
                tester.ask_question(question)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.GREEN}👋 Goodbye!")
            break
        except Exception as e:
            tester.print_error(f"Error: {str(e)}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            run_comprehensive_test()
        elif command == 'interactive' or command == 'i':
            interactive_mode()
        elif command == 'health':
            tester = AgentTester()
            tester.test_health()
        elif command == 'info':
            tester = AgentTester()
            tester.get_info()
        else:
            print(f"{Fore.RED}Unknown command: {command}")
            print_help()
    else:
        print_help()

def print_help():
    """Print help information"""
    print(f"{Fore.GREEN}{Style.BRIGHT}🧪 Abhay's AI Agent Test Script")
    print(f"\n{Fore.CYAN}Usage:")
    print(f"  {Fore.WHITE}python test_agent.py test{Fore.CYAN}        - Run comprehensive test suite")
    print(f"  {Fore.WHITE}python test_agent.py interactive{Fore.CYAN} - Interactive testing mode")
    print(f"  {Fore.WHITE}python test_agent.py i{Fore.CYAN}           - Interactive mode (short)")
    print(f"  {Fore.WHITE}python test_agent.py health{Fore.CYAN}      - Quick health check")
    print(f"  {Fore.WHITE}python test_agent.py info{Fore.CYAN}        - Get API information")
    
    print(f"\n{Fore.YELLOW}Examples:")
    print(f"  {Fore.WHITE}python test_agent.py test{Fore.YELLOW}       # Test all question types")
    print(f"  {Fore.WHITE}python test_agent.py interactive{Fore.YELLOW} # Chat with the agent")
    
    print(f"\n{Fore.BLUE}Make sure your FastAPI server is running on {BASE_URL}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.GREEN}👋 Goodbye!")
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {str(e)}")