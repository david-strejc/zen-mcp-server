#!/usr/bin/env python3
"""
Test script to debug Gemini API responses with repomix files.

This script tests sending a repomix file to Gemini and analyzes the response
to understand why the content might appear empty or missing.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types

from providers.gemini import GeminiModelProvider
from utils.file_utils import read_file_content, estimate_tokens
from utils.token_utils import estimate_tokens as token_estimate

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_gemini_repomix_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def test_direct_gemini_api(file_path: str, api_key: str):
    """Test direct Gemini API call with the file."""
    logger.info("=" * 80)
    logger.info("TESTING DIRECT GEMINI API")
    logger.info("=" * 80)
    
    try:
        # Read file content
        logger.info(f"Reading file: {file_path}")
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        logger.info(f"File content length: {len(file_content):,} characters")
        
        # Estimate tokens
        estimated_tokens = len(file_content) // 4  # Rough estimate
        logger.info(f"Estimated tokens: {estimated_tokens:,}")
        
        # Check first few lines to understand format
        first_lines = file_content.split('\n')[:20]
        logger.info("First 20 lines of file:")
        for i, line in enumerate(first_lines, 1):
            logger.debug(f"  Line {i}: {line[:100]}...")
        
        # Create Gemini client
        logger.info("Creating Gemini client...")
        client = genai.Client(api_key=api_key)
        
        # Create prompt
        prompt = f"""Please analyze this codebase and provide a comprehensive review.

The file contains a complete codebase in repomix format. Please:
1. Identify the main components and services
2. List any potential issues or bugs
3. Suggest improvements

Here is the codebase:

{file_content[:100000]}  # Limiting to first 100k chars for initial test

[Note: File truncated for testing. Full file is {len(file_content):,} characters]
"""
        
        logger.info(f"Prompt length: {len(prompt):,} characters")
        
        # Generate content
        logger.info("Calling Gemini API...")
        start_time = time.time()
        
        generation_config = types.GenerateContentConfig(
            temperature=0.7,
            candidate_count=1,
            max_output_tokens=8192,
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[{"parts": [{"text": prompt}]}],
            config=generation_config,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"API call completed in {elapsed:.2f} seconds")
        
        # Analyze response
        logger.info("=" * 80)
        logger.info("RESPONSE ANALYSIS")
        logger.info("=" * 80)
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response object: {response}")
        
        if hasattr(response, '__dict__'):
            logger.info("Response attributes:")
            for key, value in response.__dict__.items():
                logger.debug(f"  {key}: {type(value)} = {str(value)[:200]}...")
        
        # Check candidates
        if hasattr(response, 'candidates'):
            logger.info(f"Candidates count: {len(response.candidates) if response.candidates else 0}")
            if response.candidates:
                for i, candidate in enumerate(response.candidates):
                    logger.info(f"Candidate {i}:")
                    logger.debug(f"  Type: {type(candidate)}")
                    logger.debug(f"  Content: {candidate}")
                    if hasattr(candidate, 'finish_reason'):
                        logger.debug(f"  Finish reason: {candidate.finish_reason}")
        
        # Try to extract text
        logger.info("Attempting to extract text...")
        try:
            text = response.text
            if text:
                logger.info(f"✅ Successfully extracted text")
                logger.info(f"Text length: {len(text):,} characters")
                logger.info(f"First 500 characters of response:")
                logger.info(text[:500])
                logger.info("..." if len(text) > 500 else "[End of response]")
                
                # Check if response mentions the codebase
                if "service" in text.lower() or "class" in text.lower() or "function" in text.lower():
                    logger.info("✅ Response appears to reference code elements")
                else:
                    logger.warning("⚠️ Response may not be analyzing the code")
            else:
                logger.warning("❌ No text in response!")
        except Exception as e:
            logger.error(f"❌ Error extracting text: {e}")
            logger.exception(e)
        
        # Check usage metadata
        if hasattr(response, 'usage_metadata'):
            logger.info("Usage metadata:")
            metadata = response.usage_metadata
            if hasattr(metadata, 'prompt_token_count'):
                logger.info(f"  Input tokens: {metadata.prompt_token_count:,}")
            if hasattr(metadata, 'candidates_token_count'):
                logger.info(f"  Output tokens: {metadata.candidates_token_count:,}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in direct API test: {e}")
        logger.exception(e)
        return None


def test_with_provider(file_path: str, api_key: str):
    """Test using our GeminiModelProvider."""
    logger.info("=" * 80)
    logger.info("TESTING WITH GEMINI PROVIDER")
    logger.info("=" * 80)
    
    try:
        # Initialize provider
        provider = GeminiModelProvider(api_key)
        
        # Read file using our utilities
        logger.info(f"Reading file with our utilities: {file_path}")
        file_content, token_count = read_file_content(file_path, max_size=2_000_000)
        logger.info(f"File content from utils: {len(file_content):,} chars, {token_count:,} estimated tokens")
        
        # Create prompt
        prompt = f"""Please analyze this codebase:

{file_content}

Provide a brief analysis of:
1. Main components
2. Code quality
3. Potential issues"""
        
        logger.info(f"Calling provider.generate_content...")
        
        # Generate content
        response = provider.generate_content(
            prompt=prompt,
            model_name="gemini-2.5-pro",
            temperature=0.7,
            thinking_mode="medium"
        )
        
        logger.info("=" * 80)
        logger.info("PROVIDER RESPONSE ANALYSIS")
        logger.info("=" * 80)
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response content length: {len(response.content) if response.content else 0}")
        
        if response.content:
            logger.info(f"✅ Got response content")
            logger.info(f"First 500 chars:")
            logger.info(response.content[:500])
        else:
            logger.warning("❌ No content in provider response")
        
        if response.usage:
            logger.info(f"Token usage: {response.usage}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in provider test: {e}")
        logger.exception(e)
        return None


def test_file_format_analysis(file_path: str):
    """Analyze the repomix file format."""
    logger.info("=" * 80)
    logger.info("FILE FORMAT ANALYSIS")
    logger.info("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it's XML format
        if content.strip().startswith('<?php'):
            logger.info("File starts with PHP tag")
        elif content.strip().startswith('<'):
            logger.info("File appears to be XML format")
            
            # Count file entries
            file_count = content.count('<file path=')
            logger.info(f"Found {file_count} <file> entries in repomix")
            
            # Find some example files
            import re
            files = re.findall(r'<file path="([^"]+)">', content[:50000])  # Check first 50k chars
            logger.info(f"Example files found:")
            for i, file in enumerate(files[:10]):
                logger.info(f"  {i+1}. {file}")
        else:
            logger.info("File format unclear, first 200 chars:")
            logger.info(content[:200])
        
        # Check for actual PHP/code content
        has_php_code = '<?php' in content
        has_namespace = 'namespace ' in content
        has_class = 'class ' in content
        has_function = 'function ' in content
        
        logger.info(f"Content indicators:")
        logger.info(f"  Has PHP tags: {has_php_code}")
        logger.info(f"  Has namespace declarations: {has_namespace}")
        logger.info(f"  Has class definitions: {has_class}")
        logger.info(f"  Has function definitions: {has_function}")
        
    except Exception as e:
        logger.error(f"Error analyzing file format: {e}")


def main():
    """Main test function."""
    # Configuration
    file_path = "/home/david/Work/Programming/outlookint/outlook-integration-new/repomix-output.php"
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No GOOGLE_API_KEY or GEMINI_API_KEY found in environment")
        return 1
    
    logger.info(f"Using API key: {api_key[:10]}...")
    
    # Check file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    # Run tests
    logger.info("Starting Gemini repomix debug tests...")
    logger.info(f"Target file: {file_path}")
    
    # 1. Analyze file format
    test_file_format_analysis(file_path)
    
    # 2. Test direct API
    logger.info("\n" + "=" * 80)
    direct_response = test_direct_gemini_api(file_path, api_key)
    
    # 3. Test with provider
    logger.info("\n" + "=" * 80)
    provider_response = test_with_provider(file_path, api_key)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    if direct_response:
        try:
            direct_text = direct_response.text
            logger.info(f"✅ Direct API: Got {len(direct_text):,} chars response")
        except:
            logger.info("❌ Direct API: Failed to get text")
    else:
        logger.info("❌ Direct API: No response")
    
    if provider_response and provider_response.content:
        logger.info(f"✅ Provider: Got {len(provider_response.content):,} chars response")
    else:
        logger.info("❌ Provider: No content in response")
    
    logger.info("\nCheck 'test_gemini_repomix_debug.log' for full details")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())