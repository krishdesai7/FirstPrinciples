#!/usr/bin/env python3
"""
Diagnostic script to examine TPBench dataset structure and content
"""

from datasets import load_dataset
import json

def examine_tpbench():
    """Examine TPBench dataset structure and sample content"""
    print("Loading TPBench dataset...")
    dataset = load_dataset("ZhiqiGao/TPBench")
    
    # Dataset structure
    print("\n=== Dataset Structure ===")
    print(f"Type: {type(dataset)}")
    print(f"Available splits: {list(dataset.keys())}")
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  - Number of examples: {len(split_data)}")
        print(f"  - Features: {split_data.features}")
        print(f"  - Column names: {split_data.column_names}")
    
    # Sample content examination
    print("\n=== Sample Content Analysis ===")
    if len(dataset) > 0:
        first_split = list(dataset.keys())[0]
        sample = dataset[first_split][0]
        
        print(f"\nFirst example from '{first_split}' split:")
        for key, value in sample.items():
            print(f"\n{key}:")
            if isinstance(value, str):
                print(f"  Length: {len(value)} characters")
                print(f"  Preview: {value[:200]}..." if len(value) > 200 else f"  Content: {value}")
            else:
                print(f"  Value: {value}")
                print(f"  Type: {type(value)}")
    
    # Physics content analysis
    print("\n=== Physics Content Analysis ===")
    first_split = list(dataset.keys())[0]
    problems = dataset[first_split]['problem'][:5] if 'problem' in dataset[first_split].column_names else []
    
    physics_domains = {
        'mechanics': ['force', 'momentum', 'energy', 'velocity', 'acceleration'],
        'electromagnetism': ['electric', 'magnetic', 'field', 'charge', 'current'],
        'quantum': ['quantum', 'wave function', 'uncertainty', 'eigenstate'],
        'thermodynamics': ['temperature', 'entropy', 'heat', 'thermal'],
        'relativity': ['relativity', 'spacetime', 'lorentz', 'einstein']
    }
    
    domain_counts = {domain: 0 for domain in physics_domains}
    
    for problem in problems[:20]:  # Analyze first 20 problems
        problem_lower = problem.lower() if isinstance(problem, str) else ""
        for domain, keywords in physics_domains.items():
            if any(keyword in problem_lower for keyword in keywords):
                domain_counts[domain] += 1
    
    print("Physics domains in first 20 problems:")
    for domain, count in domain_counts.items():
        print(f"  - {domain}: {count} problems")

if __name__ == "__main__":
    examine_tpbench()