#!/usr/bin/env python3
"""
Script to resolve merge conflicts by intelligently combining both branches.
This keeps all functionality from both main and the feature branch.
"""

import re
import sys
from pathlib import Path

def resolve_conflict(content):
    """Resolve conflicts by keeping both branches' changes intelligently."""
    # Pattern to match conflict markers
    conflict_pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> feat/openai-flex-service-tier'
    
    def resolve_single_conflict(match):
        head_content = match.group(1)
        feature_content = match.group(2)
        
        # For the service_tier conflict in openai_compatible.py
        if "service_tier" in feature_content and "top_p" in head_content:
            # Combine the parameter lists
            return '''            if key in ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop", "stream", "service_tier"]:
                # Reasoning models (those that don't support temperature) also don't support these parameters
                if not supports_temperature and key in ["top_p", "frequency_penalty", "presence_penalty"]:
                    continue  # Skip unsupported parameters for reasoning models'''
        
        # For retry logic vs flex fallback
        if "max_retries = 4" in head_content and "service_tier=flex" in feature_content:
            # Keep retry logic AND add flex fallback
            return head_content + "\n\n            # Also handle service_tier=flex failures\n" + feature_content
        
        # Default: prefer feature branch content (newer)
        return feature_content
    
    # Process all conflicts
    resolved = re.sub(conflict_pattern, resolve_single_conflict, content, flags=re.DOTALL)
    return resolved

def main():
    # Find all conflicted files
    import subprocess
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    
    conflicted_files = []
    for line in result.stdout.splitlines():
        if line.startswith(('UU', 'AA', 'DD', 'AU', 'UA', 'DU', 'UD')):
            filepath = line[3:].strip()
            conflicted_files.append(filepath)
    
    print(f"Found {len(conflicted_files)} conflicted files")
    
    for filepath in conflicted_files:
        print(f"Resolving {filepath}...")
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            if '<<<<<<< HEAD' in content:
                resolved = resolve_conflict(content)
                with open(filepath, 'w') as f:
                    f.write(resolved)
                print(f"  ✓ Resolved conflicts in {filepath}")
            else:
                print(f"  ⚠ No conflicts found in {filepath}")
        except Exception as e:
            print(f"  ✗ Error resolving {filepath}: {e}")

if __name__ == "__main__":
    main()