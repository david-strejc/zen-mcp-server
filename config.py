"""
Configuration and constants for Zen MCP Server

This module centralizes all configuration settings for the Zen MCP Server.
It defines model configurations, token limits, temperature defaults, and other
constants used throughout the application.

Configuration values can be overridden by environment variables where appropriate.
"""

import os

# Version and metadata
# These values are used in server responses and for tracking releases
# IMPORTANT: This is the single source of truth for version and author info
# Semantic versioning: MAJOR.MINOR.PATCH
__version__ = "5.8.3"
# Last update date in ISO format
__updated__ = "2025-08-25"
# Primary maintainer
__author__ = "Fahad Gilani, David Strejc"

# Model configuration
# DEFAULT_MODEL: The default model used for all AI operations
# This should be a stable, high-performance model suitable for code analysis
# Can be overridden by setting DEFAULT_MODEL environment variable
# Special value "auto" means Claude should pick the best model for each task
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "auto")

# Auto mode detection - when DEFAULT_MODEL is "auto", Claude picks the model
IS_AUTO_MODE = DEFAULT_MODEL.lower() == "auto"

# Each provider (gemini.py, openai_provider.py, xai.py) defines its own SUPPORTED_MODELS
# with detailed descriptions. Tools use ModelProviderRegistry.get_available_model_names()
# to get models only from enabled providers (those with valid API keys).
#
# This architecture ensures:
# - No namespace collisions (models only appear when their provider is enabled)
# - API key-based filtering (prevents wrong models from being shown to Claude)
# - Proper provider routing (models route to the correct API endpoint)
# - Clean separation of concerns (providers own their model definitions)


# Temperature defaults for different tool types
# Temperature controls the randomness/creativity of model responses
# Lower values (0.0-0.3) produce more deterministic, focused responses
# Higher values (0.7-1.0) produce more creative, varied responses

# TEMPERATURE_ANALYTICAL: Used for tasks requiring precision and consistency
# Ideal for code review, debugging, and error analysis where accuracy is critical
TEMPERATURE_ANALYTICAL = 0.2  # For code review, debugging

# TEMPERATURE_BALANCED: Middle ground for general conversations
# Provides a good balance between consistency and helpful variety
TEMPERATURE_BALANCED = 0.5  # For general chat

# TEMPERATURE_CREATIVE: Higher temperature for exploratory tasks
# Used when brainstorming, exploring alternatives, or architectural discussions
TEMPERATURE_CREATIVE = 0.7  # For architecture, deep thinking

# Thinking Mode Defaults
# DEFAULT_THINKING_MODE_THINKDEEP: Default thinking depth for extended reasoning tool
# Higher modes use more computational budget but provide deeper analysis
DEFAULT_THINKING_MODE_THINKDEEP = os.getenv("DEFAULT_THINKING_MODE_THINKDEEP", "high")

# Consensus Tool Defaults
# Consensus timeout and rate limiting settings
DEFAULT_CONSENSUS_TIMEOUT = 120.0  # 2 minutes per model
DEFAULT_CONSENSUS_MAX_INSTANCES_PER_COMBINATION = 2

# NOTE: Consensus tool now uses sequential processing for MCP compatibility
# Concurrent processing was removed to avoid async pattern violations

# MCP Protocol Transport Limits
#
# IMPORTANT: This limit ONLY applies to the Claude CLI ↔ MCP Server transport boundary.
# It does NOT limit internal MCP Server operations like system prompts, file embeddings,
# conversation history, or content sent to external models (Gemini/O3/OpenRouter).
#
# MCP Protocol Architecture:
# Claude CLI ←→ MCP Server ←→ External Model (Gemini/O3/etc.)
#     ↑                              ↑
#     │                              │
# MCP transport                Internal processing
# (token limit from MAX_MCP_OUTPUT_TOKENS)    (No MCP limit - can be 1M+ tokens)
#
# MCP_PROMPT_SIZE_LIMIT: Maximum character size for USER INPUT crossing MCP transport
# The MCP protocol has a combined request+response limit controlled by MAX_MCP_OUTPUT_TOKENS.
# To ensure adequate space for MCP Server → Claude CLI responses, we limit user input
# to roughly 60% of the total token budget converted to characters. Larger user prompts
# must be sent as prompt.txt files to bypass MCP's transport constraints.
#
# Token to character conversion ratio: ~4 characters per token (average for code/text)
# Default allocation: 60% of tokens for input, 40% for response
#
# What IS limited by this constant:
# - request.prompt field content (user input from Claude CLI)
# - prompt.txt file content (alternative user input method)
# - Any other direct user input fields
#
# What is NOT limited by this constant:
# - System prompts added internally by tools
# - File content embedded by tools
# - Conversation history loaded from storage
# - Web search instructions or other internal additions
# - Complete prompts sent to external models (managed by model-specific token limits)
#
# This ensures MCP transport stays within protocol limits while allowing internal
# processing to use full model context windows (200K-1M+ tokens).


def _calculate_mcp_prompt_limit() -> int:
    """
    Calculate MCP prompt size limit based on MAX_MCP_OUTPUT_TOKENS environment variable.

    Returns:
        Maximum character count for user input prompts
    """
    # Check for Claude's MAX_MCP_OUTPUT_TOKENS environment variable
    max_tokens_str = os.getenv("MAX_MCP_OUTPUT_TOKENS")

    if max_tokens_str:
        try:
            max_tokens = int(max_tokens_str)
            # Allocate 60% of tokens for input, convert to characters (~4 chars per token)
            input_token_budget = int(max_tokens * 0.6)
            character_limit = input_token_budget * 4
            return character_limit
        except (ValueError, TypeError):
            # Fall back to default if MAX_MCP_OUTPUT_TOKENS is not a valid integer
            pass

    # Default fallback: 60,000 characters (equivalent to ~15k tokens input of 25k total)
    return 60_000


MCP_PROMPT_SIZE_LIMIT = _calculate_mcp_prompt_limit()

# Language/Locale Configuration
# LOCALE: Language/locale specification for AI responses
# When set, all AI tools will respond in the specified language while
# maintaining their analytical capabilities
# Examples: "fr-FR", "en-US", "zh-CN", "zh-TW", "ja-JP", "ko-KR", "es-ES",
# "de-DE", "it-IT", "pt-PT"
# Leave empty for default language (English)
LOCALE = os.getenv("LOCALE", "")

# OpenAI Flex Processing configuration
# When enabled (default), automatically uses OpenAI's Flex Processing service tier
# for o3 and o3-mini models to reduce costs by ~50% with slightly higher latency
# Set to "0" or "false" to disable and use standard tier
OPENAI_USE_FLEX_PROCESSING = os.getenv("OPENAI_USE_FLEX_PROCESSING", "1").lower() not in ["0", "false", "no"]

# Model capabilities descriptions
# This dictionary provides human-readable descriptions of each model's capabilities
# Used in the model selection UI when DEFAULT_MODEL is set to "auto"
MODEL_CAPABILITIES_DESC = {
    # Gemini models
    "gemini-2.0-flash": "Fast, versatile model for general tasks with solid reasoning",
    "gemini-2.0-flash-lite": "Ultrafast, lightweight model for simple tasks",
    "gemini-2.5-flash": "Ultra-fast model for quick analysis and rapid iterations",
    "gemini-2.5-pro": "EXTREMELY large context - best for consulting many files",
    # OpenAI o3 models
    "o3": "SMARTEST colleague for consulting algorithms",
    "o3-mini": "Balanced reasoning model with good speed/quality trade-off",
    # OpenAI GPT-5 models
    "gpt-5": "Use if everything else is failing or for multi-consultations",
    "gpt-5-mini": "Balanced GPT-5 variant with good performance/cost ratio",
    "gpt-5-nano": "Fastest and most cost-effective GPT-5 variant",
    # XAI Grok models
    "grok-3": "State-of-the-art model with advanced capabilities",
    "grok-3-fast": "Fast version of Grok-3 for quick responses",
    # Aliases for convenience
    "pro": "gemini-2.5-pro",
    "flash": "gemini-2.5-flash",
    "flashlite": "gemini-2.0-flash-lite",
    "lite": "gemini-2.0-flash-lite",
    "gpt5": "gpt-5",
    "gpt5-mini": "gpt-5-mini",
    "gpt5mini": "gpt-5-mini",
    "gpt5-nano": "gpt-5-nano",
    "gpt5nano": "gpt-5-nano",
    "nano": "gpt-5-nano",
}

# Threading configuration
# Simple in-memory conversation threading for stateless MCP environment
# Conversations persist only during the Claude session
