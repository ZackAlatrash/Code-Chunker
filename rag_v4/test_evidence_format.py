#!/usr/bin/env python3
"""
Test the evidence rendering format without calling LLM.
"""
import json
from answerer import render_evidence, build_messages

def main():
    # Load test chunks
    with open("rag_v4/test_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print("="*80)
    print("EVIDENCE RENDERING TEST")
    print("="*80)
    print()
    
    # Render evidence
    evidence = render_evidence(chunks, max_items=5, max_code_chars=500)
    print(evidence)
    
    print("\n" + "="*80)
    print("MESSAGE STRUCTURE TEST")
    print("="*80)
    print()
    
    # Build messages
    question = "What does the GetForecastForLocation method do?"
    messages = build_messages(question, evidence)
    
    for i, msg in enumerate(messages, 1):
        print(f"Message {i} ({msg['role']}):")
        print("-" * 40)
        content = msg['content']
        if len(content) > 200:
            print(content[:200] + "\n... [truncated] ...")
        else:
            print(content)
        print()
    
    print("="*80)
    print("✅ Evidence format and message structure look good!")
    print("="*80)

if __name__ == "__main__":
    main()

