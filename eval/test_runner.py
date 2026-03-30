"""
Evaluation Suite Runner - Phase 3
Iterates over test cases and passes them to the Ecommerce Support Crew.
"""
import os
import json
import time
from dotenv import load_dotenv

# Load env immediately so imports downstream capture keys
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# We must adjust the system path so we can import 'main' and 'src' 
# if we're running this from inside the 'eval' directory.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_ecommerce_support_crew

TEST_CASES_FILE = os.path.join(os.path.dirname(__file__), "test_cases.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "eval_results.json")

def main():
    if not os.path.exists(TEST_CASES_FILE):
        print(f"Test cases file missing: {TEST_CASES_FILE}")
        return
        
    with open(TEST_CASES_FILE, "r") as f:
        cases = json.load(f)
        
    results = []
    
    print(f"Starting execution of {len(cases)} test cases...")
    for idx, case in enumerate(cases):
        print(f"\n[{idx+1}/{len(cases)}] Running Case {case['id']} ({case['type']})")
        
        try:
            raw_output = run_ecommerce_support_crew(case["ticket_text"], case["order_context_json"])
            
            # Extract output exactly as main.py does
            output_json = None
            if hasattr(raw_output, "model_dump"):
                try:
                    output_json = raw_output.model_dump()
                except:
                    pass
            elif hasattr(raw_output, "pydantic") and raw_output.pydantic:
                try:
                    output_json = raw_output.pydantic.model_dump()
                except:
                    pass
            elif isinstance(raw_output, str):
                try:
                    output_json = json.loads(raw_output)
                except Exception:
                    output_json = {"raw_string": raw_output}
                    
            if not output_json:
                output_json = {"raw_output": str(raw_output)}
                
            results.append({
                "id": case["id"],
                "type": case["type"],
                "ticket_text": case["ticket_text"],
                "success": True,
                "agent_output": output_json
            })
            
        except Exception as e:
            print(f"Error executing case {case['id']}: {e}")
            results.append({
                "id": case["id"],
                "type": case["type"],
                "success": False,
                "error": str(e) # Catch failures like Rate Limits or LLM crashes
            })
            
        print("Sleeping for 65 seconds to guarantee a completely fresh TPM bucket reset...")
        time.sleep(65)
        
    print(f"\nSaving results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
