import json
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "eval_results.json")

def calculate_metrics():
    if not os.path.exists(RESULTS_FILE):
        print(f"No results file found at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    total_runs = len(results)
    if total_runs == 0:
        print("Empty results file.")
        return

    pydantic_success = 0
    citation_coverage = 0
    
    conflict_oob_cases = 0
    escalation_success = 0

    for r in results:
        if not r.get("success"):
            continue

        agent_output = r.get("agent_output", {})
        
        # The new Minimalist matrix might output JSON embedded in a "raw" string key
        if "raw" in agent_output and isinstance(agent_output["raw"], str):
            try:
                parsed_raw = json.loads(agent_output["raw"])
                agent_output.update(parsed_raw)
            except:
                pass

        # Check Pydantic validation by hunting for expected schema keys
        if "decision" in agent_output and "rationale" in agent_output:
            pydantic_success += 1
            
        # Citation coverage
        citations = agent_output.get("citations", [])
        if isinstance(citations, list) and len(citations) > 0:
            citation_coverage += 1

        # Escalation Rate check for specific trick cases
        case_type = r.get("type", "")
        if case_type in ["Conflict", "Out-of-Scope"]:
            conflict_oob_cases += 1
            decision = agent_output.get("decision", "").lower()
            if "escalate" in decision or "abstain" in decision:
                escalation_success += 1

    print("\n--- Phase 3 Evaluation Metrics ---")
    print(f"Total Cases Run: {total_runs}")
    print(f"Pydantic Schema Parse Rate: {pydantic_success}/{total_runs} ({(pydantic_success/total_runs)*100:.1f}%)")
    print(f"Citation Coverage Rate (Excluding Failures): {citation_coverage}/{total_runs} ({(citation_coverage/total_runs)*100:.1f}%)")
    
    if conflict_oob_cases > 0:
        print(f"Escalation Rate (Defensive Metric): {escalation_success}/{conflict_oob_cases} ({(escalation_success/conflict_oob_cases)*100:.1f}%)")
    else:
        print("No Conflict/OOB cases were tested for Escalation metrics.")

if __name__ == "__main__":
    calculate_metrics()
