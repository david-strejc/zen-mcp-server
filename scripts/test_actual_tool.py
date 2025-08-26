#!/usr/bin/env python3
"""
Test script that uses the ACTUAL MCP tool code to debug the repomix issue.

This mimics exactly what happens when the MCP tool is called, using the same
code paths to identify where the problem occurs.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the ACTUAL tools from our codebase
from tools.analyze import AnalyzeTool
from tools.thinkdeep import ThinkDeepTool
from tools.codereview import CodeReviewTool
from tools.chat import ChatTool

# Import utilities
from utils.file_utils import read_file_content, read_files, estimate_tokens
from providers.registry import ModelProviderRegistry

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_actual_tool_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Silence some noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def test_file_reading(file_path: str):
    """Test our file reading utilities."""
    logger.info("=" * 80)
    logger.info("TESTING FILE READING UTILITIES")
    logger.info("=" * 80)
    
    # Test 1: Basic file stats
    file_size = os.path.getsize(file_path)
    logger.info(f"File path: {file_path}")
    logger.info(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # Test 2: Read with our utility (default 5MB limit now)
    logger.info("\nTesting read_file_content()...")
    content, token_count = read_file_content(file_path)
    logger.info(f"  Content length: {len(content):,} chars")
    logger.info(f"  Estimated tokens: {token_count:,}")
    
    # Check if content was truncated
    if "FILE TOO LARGE" in content:
        logger.warning("  ⚠️ File was truncated as too large!")
    else:
        logger.info("  ✅ File read successfully without truncation")
    
    # Test 3: Read with explicit size limit
    logger.info("\nTesting with 10MB limit...")
    content_10mb, tokens_10mb = read_file_content(file_path, max_size=10_000_000)
    logger.info(f"  Content length: {len(content_10mb):,} chars")
    logger.info(f"  Estimated tokens: {tokens_10mb:,}")
    
    # Test 4: Using read_files (batch reading)
    logger.info("\nTesting read_files() batch function...")
    files_content = read_files([file_path], max_tokens=500_000)
    logger.info(f"  Batch content length: {len(files_content):,} chars")
    
    return content, token_count


def test_analyze_tool(file_path: str):
    """Test the AnalyzeTool with the file."""
    logger.info("=" * 80)
    logger.info("TESTING ANALYZE TOOL")
    logger.info("=" * 80)
    
    try:
        # Create the tool instance
        tool = AnalyzeTool()
        
        # Prepare arguments exactly as MCP would send them
        arguments = {
            "model": "gemini-2.5-pro",
            "files": [file_path],
            "prompt": "Please analyze this EspoCRM Outlook Integration codebase. Focus on code quality, architecture, potential issues, and areas for improvement.",
            "analysis_type": "general",
            "thinking_mode": "high",
            "use_websearch": False,
            "output_format": "detailed"
        }
        
        logger.info(f"Calling AnalyzeTool.execute() with arguments:")
        for key, value in arguments.items():
            if key == "files":
                logger.info(f"  {key}: {value}")
            elif key == "prompt":
                logger.info(f"  {key}: {value[:100]}...")
            else:
                logger.info(f"  {key}: {value}")
        
        # Execute the tool
        logger.info("\nExecuting tool...")
        result = tool.execute(**arguments)
        
        logger.info("\n" + "=" * 40)
        logger.info("ANALYZE TOOL RESULT")
        logger.info("=" * 40)
        logger.info(f"Result type: {type(result)}")
        
        if hasattr(result, '__dict__'):
            for key, value in result.__dict__.items():
                if key == 'content':
                    logger.info(f"  {key}: {len(value) if value else 0} chars")
                    if value:
                        logger.info(f"    First 500 chars: {value[:500]}...")
                else:
                    logger.info(f"  {key}: {str(value)[:200]}...")
        
        if result and hasattr(result, 'content'):
            if result.content:
                logger.info(f"✅ Got response: {len(result.content):,} chars")
                
                # Check if it's the "files required" message
                if "files_required_to_continue" in result.content:
                    logger.warning("❌ Tool thinks files are missing!")
                    logger.warning(f"Response: {result.content[:500]}")
            else:
                logger.warning("❌ No content in result!")
        else:
            logger.warning("❌ Invalid result structure!")
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing AnalyzeTool: {e}")
        logger.exception(e)
        return None


def test_thinkdeep_tool(file_path: str):
    """Test the ThinkDeepTool with the file."""
    logger.info("=" * 80)
    logger.info("TESTING THINKDEEP TOOL")
    logger.info("=" * 80)
    
    try:
        # Create the tool instance
        tool = ThinkDeepTool()
        
        # Prepare arguments
        arguments = {
            "model": "gemini-2.5-pro",
            "prompt": "Analyze this codebase and tell me what it does",
            "files": [file_path],
            "thinking_mode": "medium",
            "use_websearch": False,
        }
        
        logger.info(f"Calling ThinkDeepTool.execute()...")
        
        # Execute the tool
        result = tool.execute(**arguments)
        
        logger.info("\n" + "=" * 40)
        logger.info("THINKDEEP TOOL RESULT")
        logger.info("=" * 40)
        
        if result and hasattr(result, 'content'):
            if result.content:
                logger.info(f"✅ Got response: {len(result.content):,} chars")
                logger.info(f"First 300 chars: {result.content[:300]}...")
            else:
                logger.warning("❌ No content in result!")
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing ThinkDeepTool: {e}")
        logger.exception(e)
        return None


def test_file_preparation(file_path: str):
    """Test the _prepare_file_content_for_prompt method."""
    logger.info("=" * 80)
    logger.info("TESTING FILE PREPARATION IN BASE TOOL")
    logger.info("=" * 80)
    
    try:
        # Create a tool instance to access the base methods
        tool = AnalyzeTool()
        
        # Test the file preparation method directly
        logger.info("Testing _prepare_file_content_for_prompt()...")
        
        # Mock model context
        from utils.model_context import ModelContext
        model_context = ModelContext("gemini-2.5-pro")
        tool._model_context = model_context
        
        # Call the method that prepares files
        file_content, processed_files, file_refs = tool._prepare_file_content_for_prompt(
            request_files=[file_path],
            continuation_id=None,
            context_description="Test files",
            max_tokens=1_000_000,
            reserve_tokens=50_000,
            model_context=model_context
        )
        
        logger.info(f"Result:")
        logger.info(f"  File content length: {len(file_content) if file_content else 0} chars")
        logger.info(f"  Processed files: {processed_files}")
        logger.info(f"  File references: {file_refs}")
        
        if file_content:
            logger.info(f"  First 500 chars: {file_content[:500]}...")
            
            # Check for truncation messages
            if "FILE TOO LARGE" in file_content:
                logger.warning("  ⚠️ File marked as too large!")
            elif "SKIPPED FILES" in file_content:
                logger.warning("  ⚠️ Files were skipped!")
            else:
                logger.info("  ✅ File content prepared successfully")
        else:
            logger.warning("  ❌ No file content returned!")
        
        return file_content
        
    except Exception as e:
        logger.error(f"Error in file preparation test: {e}")
        logger.exception(e)
        return None


def test_provider_directly(file_path: str):
    """Test calling the Gemini provider directly with file content."""
    logger.info("=" * 80)
    logger.info("TESTING GEMINI PROVIDER DIRECTLY")
    logger.info("=" * 80)
    
    try:
        from providers.gemini import GeminiModelProvider
        
        # Get API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("No API key found!")
            return None
        
        # Read file
        content, tokens = read_file_content(file_path, max_size=10_000_000)
        logger.info(f"File content: {len(content):,} chars, {tokens:,} tokens")
        
        # Create provider
        provider = GeminiModelProvider(api_key)
        
        # Create simple prompt
        prompt = f"""Analyze this codebase:

{content}

What are the main components?"""
        
        logger.info(f"Prompt length: {len(prompt):,} chars")
        logger.info("Calling provider.generate_content()...")
        
        # Call provider
        response = provider.generate_content(
            prompt=prompt,
            model_name="gemini-2.5-pro",
            temperature=0.7,
            thinking_mode="medium"
        )
        
        logger.info(f"Response type: {type(response)}")
        if response and response.content:
            logger.info(f"✅ Got response: {len(response.content):,} chars")
            logger.info(f"First 500 chars: {response.content[:500]}...")
        else:
            logger.warning("❌ No content in response!")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calling provider: {e}")
        logger.exception(e)
        return None


def main():
    """Main test function."""
    file_path = "/home/david/Work/Programming/outlookint/outlook-integration-new/repomix-output.php"
    
    # Check file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    logger.info("Starting comprehensive MCP tool debugging...")
    logger.info(f"Target file: {file_path}")
    logger.info("")
    
    # Test 1: File reading
    file_content, tokens = test_file_reading(file_path)
    
    # Test 2: File preparation in base tool
    logger.info("\n" + "=" * 80)
    prepared_content = test_file_preparation(file_path)
    
    # Test 3: Provider directly
    logger.info("\n" + "=" * 80)
    provider_response = test_provider_directly(file_path)
    
    # Test 4: Analyze tool
    logger.info("\n" + "=" * 80)
    analyze_result = test_analyze_tool(file_path)
    
    # Test 5: ThinkDeep tool
    logger.info("\n" + "=" * 80)
    thinkdeep_result = test_thinkdeep_tool(file_path)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    logger.info("File reading:")
    if file_content and "FILE TOO LARGE" not in file_content:
        logger.info(f"  ✅ File read OK: {len(file_content):,} chars")
    else:
        logger.info(f"  ❌ File reading issue")
    
    logger.info("File preparation:")
    if prepared_content and len(prepared_content) > 1000:
        logger.info(f"  ✅ File prepared OK: {len(prepared_content):,} chars")
    else:
        logger.info(f"  ❌ File preparation issue")
    
    logger.info("Provider direct call:")
    if provider_response and provider_response.content:
        logger.info(f"  ✅ Provider OK: {len(provider_response.content):,} chars")
    else:
        logger.info(f"  ❌ Provider issue")
    
    logger.info("Analyze tool:")
    if analyze_result and analyze_result.content and "files_required" not in analyze_result.content:
        logger.info(f"  ✅ Analyze OK: {len(analyze_result.content):,} chars")
    else:
        logger.info(f"  ❌ Analyze tool issue")
    
    logger.info("ThinkDeep tool:")
    if thinkdeep_result and thinkdeep_result.content and "files_required" not in thinkdeep_result.content:
        logger.info(f"  ✅ ThinkDeep OK: {len(thinkdeep_result.content):,} chars")
    else:
        logger.info(f"  ❌ ThinkDeep tool issue")
    
    logger.info("\nCheck 'test_actual_tool_debug.log' for full details")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())