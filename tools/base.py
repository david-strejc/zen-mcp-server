"""
Base class for all Zen MCP tools

This module provides the abstract base class that all tools must inherit from.
It defines the contract that tools must implement and provides common functionality
for request validation, error handling, and response formatting.

Key responsibilities:
- Define the tool interface (abstract methods that must be implemented)
- Handle request validation and file path security
- Manage Gemini model creation with appropriate configurations
- Standardize response formatting and error handling
- Support for clarification requests when more information is needed
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Optional

from mcp.types import TextContent
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

import config
from config import MCP_PROMPT_SIZE_LIMIT
from providers import ModelProvider, ModelProviderRegistry
from utils import check_token_limit
from utils.conversation_memory import (
    MAX_CONVERSATION_TURNS,
    add_turn,
    create_thread,
    get_conversation_file_list,
    get_thread,
)
from utils.file_storage import FileReference, FileStorage
from utils.file_utils import read_file_content, read_files

from .models import SPECIAL_STATUS_MODELS, ContinuationOffer, ToolOutput

logger = logging.getLogger(__name__)


class ToolRequest(BaseModel):
    """
    Base request model for all tools.

    This Pydantic model defines common parameters that can be used by any tool.
    Tools can extend this model to add their specific parameters while inheriting
    these common fields.
    """

    model: Optional[str] = Field(
        None,
        description="Model to use. See tool's input schema for available models and their capabilities.",
    )
    temperature: Optional[float] = Field(None, description="Temperature for response (tool-specific defaults)")
    # Thinking mode controls how much computational budget the model uses for reasoning
    # Higher values allow for more complex reasoning but increase latency and cost
    thinking_mode: Optional[Literal["minimal", "low", "medium", "high", "max"]] = Field(
        None,
        description=(
            "Thinking depth: minimal (0.5% of model max), low (8%), medium (33%), high (67%), max (100% of model max)"
        ),
    )
    use_websearch: Optional[bool] = Field(
        True,
        description=(
            "Enable web search for documentation, best practices, and current information. "
            "When enabled, the model can request Claude to perform web searches and share results back "
            "during conversations. Particularly useful for: brainstorming sessions, architectural design "
            "discussions, exploring industry best practices, working with specific frameworks/technologies, "
            "researching solutions to complex problems, or when current documentation and community insights "
            "would enhance the analysis."
        ),
    )
    continuation_id: Optional[str] = Field(
        None,
        description=(
            "Thread continuation ID for multi-turn conversations. When provided, the complete conversation "
            "history is automatically embedded as context. Your response should build upon this history "
            "without repeating previous analysis or instructions. Focus on providing only new insights, "
            "additional findings, or answers to follow-up questions. Can be used across different tools."
        ),
    )
    file_handling_mode: Optional[Literal["embedded", "summary", "reference"]] = Field(
        "embedded",
        description=(
            "How to handle file content in responses. 'embedded' includes full content (default), "
            "'summary' returns only summaries to save tokens, 'reference' stores files and returns IDs."
        ),
    )


class BaseTool(ABC):
    """
    Abstract base class for all Gemini tools.

    This class defines the interface that all tools must implement and provides
    common functionality for request handling, model creation, and response formatting.

    To create a new tool:
    1. Create a new class that inherits from BaseTool
    2. Implement all abstract methods
    3. Define a request model that inherits from ToolRequest
    4. Register the tool in server.py's TOOLS dictionary
    """

    def __init__(self):
        # Cache tool metadata at initialization to avoid repeated calls
        self.name = self.get_name()
        self.description = self.get_description()
        self.default_temperature = self.get_default_temperature()
        # Initialize file storage
        self.file_storage = FileStorage()

    def requires_model(self) -> bool:
        """
        Return whether this tool requires AI model access.

        Tools that override execute() to do pure data processing (like planner)
        should return False to skip model resolution at the MCP boundary.

        Returns:
            True by default, False for tools that don't need model access
        """
        return True
        # Tool initialization complete

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name identifier for this tool.

        This name is used by MCP clients to invoke the tool and must be
        unique across all registered tools.

        Returns:
            str: The tool's unique name (e.g., "review_code", "analyze")
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return a detailed description of what this tool does.

        This description is shown to MCP clients (like Claude) to help them
        understand when and how to use the tool. It should be comprehensive
        and include trigger phrases.

        Returns:
            str: Detailed tool description with usage examples
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> dict[str, Any]:
        """
        Return the JSON Schema that defines this tool's parameters.

        This schema is used by MCP clients to validate inputs before
        sending requests. It should match the tool's request model.

        Returns:
            Dict[str, Any]: JSON Schema object defining required and optional parameters
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt that configures the AI model's behavior.

        This prompt sets the context and instructions for how the model
        should approach the task. It's prepended to the user's request.

        Returns:
            str: System prompt with role definition and instructions
        """
        pass

    def is_effective_auto_mode(self) -> bool:
        """
        Check if we're in effective auto mode for schema generation.

        This determines whether the model parameter should be required in the tool schema.
        Used at initialization time when schemas are generated.

        Returns:
            bool: True if model parameter should be required in the schema
        """
        from config import DEFAULT_MODEL
        from providers.registry import ModelProviderRegistry

        # Case 1: Explicit auto mode
        if DEFAULT_MODEL.lower() == "auto":
            return True

        # Case 2: Model not available (fallback to auto mode)
        if DEFAULT_MODEL.lower() != "auto":
            provider = ModelProviderRegistry.get_provider_for_model(DEFAULT_MODEL)
            if not provider:
                return True

        return False

    def _should_require_model_selection(self, model_name: str) -> bool:
        """
        Check if we should require Claude to select a model at runtime.

        This is called during request execution to determine if we need
        to return an error asking Claude to provide a model parameter.

        Args:
            model_name: The model name from the request or DEFAULT_MODEL

        Returns:
            bool: True if we should require model selection
        """
        # Case 1: Model is explicitly "auto"
        if model_name.lower() == "auto":
            return True

        # Case 2: Requested model is not available
        from providers.registry import ModelProviderRegistry

        provider = ModelProviderRegistry.get_provider_for_model(model_name)
        if not provider:
            logger = logging.getLogger(f"tools.{self.name}")
            logger.warning(f"Model '{model_name}' is not available with current API keys. Requiring model selection.")
            return True

        return False

    def _get_available_models(self) -> list[str]:
        """
        Get list of models that are actually available with current API keys.

        This respects model restrictions automatically.

        Returns:
            List of available model names
        """
        from config import MODEL_CAPABILITIES_DESC
        from providers.base import ProviderType
        from providers.registry import ModelProviderRegistry

        # Get available models from registry (respects restrictions)
        available_models_map = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        available_models = list(available_models_map.keys())

        # Add model aliases if their targets are available
        model_aliases = []
        for alias, target in MODEL_CAPABILITIES_DESC.items():
            if alias not in available_models and target in available_models:
                model_aliases.append(alias)

        available_models.extend(model_aliases)

        # Also check if OpenRouter is available (it accepts any model)
        openrouter_provider = ModelProviderRegistry.get_provider(ProviderType.OPENROUTER)
        if openrouter_provider and not available_models:
            # If only OpenRouter is available, suggest using any model through it
            available_models.append("any model via OpenRouter")

        if not available_models:
            # Check if it's due to restrictions
            from utils.model_restrictions import get_restriction_service

            restriction_service = get_restriction_service()
            restrictions = restriction_service.get_restriction_summary()

            if restrictions:
                return ["none - all models blocked by restrictions set in .env"]
            else:
                return ["none - please configure API keys"]

        return available_models

    def get_model_field_schema(self) -> dict[str, Any]:
        """
        Generate the model field schema based on auto mode configuration.

        When auto mode is enabled, the model parameter becomes required
        and includes detailed descriptions of each model's capabilities.

        Returns:
            Dict containing the model field JSON schema
        """
        import os

        from config import DEFAULT_MODEL, MODEL_CAPABILITIES_DESC

        # Check if OpenRouter is configured
        has_openrouter = bool(
            os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_API_KEY") != "your_openrouter_api_key_here"
        )

        # Use the centralized effective auto mode check
        if self.is_effective_auto_mode():
            # In auto mode, model is required and we provide detailed descriptions
            model_desc_parts = [
                "IMPORTANT: Use the model specified by the user if provided, OR select the most suitable model "
                "for this specific task based on the requirements and capabilities listed below:"
            ]
            # Only show descriptions for available models
            available_models = self._get_available_models()
            for model, desc in MODEL_CAPABILITIES_DESC.items():
                if model in available_models:
                    model_desc_parts.append(f"- '{model}': {desc}")

            if has_openrouter:
                # Add OpenRouter models with descriptions
                try:
                    import logging

                    from providers.openrouter_registry import OpenRouterModelRegistry

                    registry = OpenRouterModelRegistry()

                    # Group models by their model_name to avoid duplicates
                    seen_models = set()
                    model_configs = []

                    for alias in registry.list_aliases():
                        config = registry.resolve(alias)
                        if config and config.model_name not in seen_models:
                            seen_models.add(config.model_name)
                            model_configs.append((alias, config))

                    # Sort by context window (descending) then by alias
                    model_configs.sort(key=lambda x: (-x[1].context_window, x[0]))

                    if model_configs:
                        model_desc_parts.append("\nOpenRouter models (use these aliases):")
                        for alias, config in model_configs[:10]:  # Limit to top 10
                            # Format context window in human-readable form
                            context_tokens = config.context_window
                            if context_tokens >= 1_000_000:
                                context_str = f"{context_tokens // 1_000_000}M"
                            elif context_tokens >= 1_000:
                                context_str = f"{context_tokens // 1_000}K"
                            else:
                                context_str = str(context_tokens)

                            # Build description line
                            if config.description:
                                desc = f"- '{alias}' ({context_str} context): {config.description}"
                            else:
                                # Fallback to showing the model name if no description
                                desc = f"- '{alias}' ({context_str} context): {config.model_name}"
                            model_desc_parts.append(desc)

                        # Add note about additional models if any were cut off
                        total_models = len(model_configs)
                        if total_models > 10:
                            model_desc_parts.append(f"... and {total_models - 10} more models available")
                except Exception as e:
                    # Log for debugging but don't fail
                    import logging

                    logging.debug(f"Failed to load OpenRouter model descriptions: {e}")
                    # Fallback to simple message
                    model_desc_parts.append(
                        "\nOpenRouter models: If configured, you can also use ANY model available on OpenRouter."
                    )

            # Get only models from enabled providers
            available_models = self._get_available_models()
            return {
                "type": "string",
                "description": "\n".join(model_desc_parts),
                "enum": available_models,
            }
        else:
            # Normal mode - model is optional with default
            available_models = self._get_available_models()
            models_str = ", ".join(f"'{m}'" for m in available_models)

            description = f"Model to use. Native models: {models_str}."
            if has_openrouter:
                # Add OpenRouter aliases
                try:
                    # Import registry directly to show available aliases
                    # This works even without an API key
                    from providers.openrouter_registry import OpenRouterModelRegistry

                    registry = OpenRouterModelRegistry()
                    aliases = registry.list_aliases()

                    # Show ALL aliases from the configuration
                    if aliases:
                        # Show all aliases so Claude knows every option available
                        all_aliases = sorted(aliases)
                        alias_list = ", ".join(f"'{a}'" for a in all_aliases)
                        description += f" OpenRouter aliases: {alias_list}."
                    else:
                        description += " OpenRouter: Any model available on openrouter.ai."
                except Exception:
                    description += (
                        " OpenRouter: Any model available on openrouter.ai "
                        "(e.g., 'gpt-4', 'claude-3-opus', 'mistral-large')."
                    )
            description += f" Defaults to '{DEFAULT_MODEL}' if not specified."

            return {
                "type": "string",
                "description": description,
            }

    def get_default_temperature(self) -> float:
        """
        Return the default temperature setting for this tool.

        Override this method to set tool-specific temperature defaults.
        Lower values (0.0-0.3) for analytical tasks, higher (0.7-1.0) for creative tasks.

        Returns:
            float: Default temperature between 0.0 and 1.0
        """
        return 0.5

    def wants_line_numbers_by_default(self) -> bool:
        """
        Return whether this tool wants line numbers added to code files by default.

        By default, ALL tools get line numbers for precise code references.
        Line numbers are essential for accurate communication about code locations.

        Line numbers add ~8-10% token overhead but provide precise targeting for:
        - Code review feedback ("SQL injection on line 45")
        - Debug error locations ("Memory leak in loop at lines 123-156")
        - Test generation targets ("Generate tests for method at lines 78-95")
        - Refactoring guidance ("Extract method from lines 67-89")
        - General code discussions ("Where is X defined?" -> "Line 42")

        The only exception is when reading diffs, which have their own line markers.

        Returns:
            bool: True if line numbers should be added by default for this tool
        """
        return True  # All tools get line numbers by default for consistency

    def get_default_thinking_mode(self) -> str:
        """
        Return the default thinking mode for this tool.

        Thinking mode controls computational budget for reasoning.
        Override for tools that need more or less reasoning depth.

        Returns:
            str: One of "minimal", "low", "medium", "high", "max"
        """
        return "medium"  # Default to medium thinking for better reasoning

    def get_model_category(self) -> "ToolModelCategory":
        """
        Return the model category for this tool.

        Model category influences which model is selected in auto mode.
        Override to specify whether your tool needs extended reasoning,
        fast response, or balanced capabilities.

        Returns:
            ToolModelCategory: Category that influences model selection
        """
        from tools.models import ToolModelCategory

        return ToolModelCategory.BALANCED

    def get_conversation_embedded_files(self, continuation_id: Optional[str]) -> list[str]:
        """
        Get list of files already embedded in conversation history.

        This method returns the list of files that have already been embedded
        in the conversation history for a given continuation thread. Tools can
        use this to avoid re-embedding files that are already available in the
        conversation context.

        Args:
            continuation_id: Thread continuation ID, or None for new conversations

        Returns:
            list[str]: List of file paths already embedded in conversation history
        """
        if not continuation_id:
            # New conversation, no files embedded yet
            return []

        thread_context = get_thread(continuation_id)
        if not thread_context:
            # Thread not found, no files embedded
            return []

        embedded_files = get_conversation_file_list(thread_context)
        logger.debug(f"[FILES] {self.name}: Found {len(embedded_files)} embedded files")
        return embedded_files

    def filter_new_files(self, requested_files: list[str], continuation_id: Optional[str]) -> list[str]:
        """
        Filter out files that are already embedded in conversation history.

        This method prevents duplicate file embeddings by filtering out files that have
        already been embedded in the conversation history. This optimizes token usage
        while ensuring tools still have logical access to all requested files through
        conversation history references.

        Args:
            requested_files: List of files requested for current tool execution
            continuation_id: Thread continuation ID, or None for new conversations

        Returns:
            list[str]: List of files that need to be embedded (not already in history)
        """
        logger.debug(f"[FILES] {self.name}: Filtering {len(requested_files)} requested files")

        if not continuation_id:
            # New conversation, all files are new
            logger.debug(f"[FILES] {self.name}: New conversation, all {len(requested_files)} files are new")
            return requested_files

        try:
            embedded_files = set(self.get_conversation_embedded_files(continuation_id))
            logger.debug(f"[FILES] {self.name}: Found {len(embedded_files)} embedded files in conversation")

            # Safety check: If no files are marked as embedded but we have a continuation_id,
            # this might indicate an issue with conversation history. Be conservative.
            if not embedded_files:
                logger.debug(f"{self.name} tool: No files found in conversation history for thread {continuation_id}")
                logger.debug(
                    f"[FILES] {self.name}: No embedded files found, returning all {len(requested_files)} requested files"
                )
                return requested_files

            # Return only files that haven't been embedded yet
            new_files = [f for f in requested_files if f not in embedded_files]
            logger.debug(
                f"[FILES] {self.name}: After filtering: {len(new_files)} new files, {len(requested_files) - len(new_files)} already embedded"
            )
            logger.debug(f"[FILES] {self.name}: New files to embed: {new_files}")

            # Log filtering results for debugging
            if len(new_files) < len(requested_files):
                skipped = [f for f in requested_files if f in embedded_files]
                logger.debug(
                    f"{self.name} tool: Filtering {len(skipped)} files already in conversation history: {', '.join(skipped)}"
                )
                logger.debug(f"[FILES] {self.name}: Skipped (already embedded): {skipped}")

            return new_files

        except Exception as e:
            # If there's any issue with conversation history lookup, be conservative
            # and include all files rather than risk losing access to needed files
            logger.warning(f"{self.name} tool: Error checking conversation history for {continuation_id}: {e}")
            logger.warning(f"{self.name} tool: Including all requested files as fallback")
            logger.debug(
                f"[FILES] {self.name}: Exception in filter_new_files, returning all {len(requested_files)} files as fallback"
            )
            return requested_files

    def _prepare_file_content_for_prompt(
        self,
        request_files: list[str],
        continuation_id: Optional[str],
        context_description: str = "New files",
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 1_000,
        remaining_budget: Optional[int] = None,
        arguments: Optional[dict] = None,
        file_handling_mode: Optional[str] = None,
        model_context: Optional[Any] = None,
    ) -> tuple[str, list[str], Optional[list[FileReference]]]:
        """
        Centralized file processing for tool prompts.

        This method handles the common pattern across all tools:
        1. Filter out files already embedded in conversation history
        2. Read content of only new files
        3. Generate informative note about skipped files
        4. Handle different file handling modes (embedded, summary, reference)

        Args:
            request_files: List of files requested for current tool execution
            continuation_id: Thread continuation ID, or None for new conversations
            context_description: Description for token limit validation (e.g. "Code", "New files")
            max_tokens: Maximum tokens to use (defaults to remaining budget or model-specific content allocation)
            reserve_tokens: Tokens to reserve for additional prompt content (default 1K)
            remaining_budget: Remaining token budget after conversation history (from server.py)
            arguments: Original tool arguments (used to extract _remaining_tokens if available)
            file_handling_mode: How to handle files - "embedded" (default), "summary", or "reference"

        Returns:
            tuple[str, list[str], Optional[list[FileReference]]]:
                - formatted_file_content: Formatted file content string ready for prompt inclusion
                - actually_processed_files: List of individual file paths that were actually read and embedded
                  (directories are expanded to individual files)
                - file_references: List of FileReference objects if mode is "summary" or "reference"

        Note: This method now separates image files and stores them in self._current_images
              for use by the execute method when calling generate_content
        """
        if not request_files:
            return "", [], None

        # Note: Even if conversation history is already embedded, we still need to process
        # any NEW files that aren't in the conversation history yet. The filter_new_files
        # method will correctly identify which files need to be embedded.

        # Extract remaining budget from arguments if available
        if remaining_budget is None:
            # Use provided arguments or fall back to stored arguments from execute()
            args_to_use = arguments or getattr(self, "_current_arguments", {})
            remaining_budget = args_to_use.get("_remaining_tokens")

        # Use remaining budget if provided, otherwise fall back to max_tokens or model-specific default
        if remaining_budget is not None:
            effective_max_tokens = remaining_budget - reserve_tokens
        elif max_tokens is not None:
            effective_max_tokens = max_tokens - reserve_tokens
        else:
            # Get model-specific limits
            # First check if model_context was passed from server.py
            model_context = None
            if arguments:
                model_context = arguments.get("_model_context") or getattr(self, "_current_arguments", {}).get(
                    "_model_context"
                )

            if model_context:
                # Use the passed model context
                try:
                    token_allocation = model_context.calculate_token_allocation()
                    effective_max_tokens = token_allocation.file_tokens - reserve_tokens
                    logger.debug(
                        f"[FILES] {self.name}: Using passed model context for {model_context.model_name}: "
                        f"{token_allocation.file_tokens:,} file tokens from {token_allocation.total_tokens:,} total"
                    )
                except Exception as e:
                    logger.warning(f"[FILES] {self.name}: Error using passed model context: {e}")
                    # Fall through to manual calculation
                    model_context = None

            if not model_context:
                # Manual calculation as fallback
                from config import DEFAULT_MODEL

                model_name = getattr(self, "_current_model_name", None) or DEFAULT_MODEL

                # Handle auto mode gracefully
                if model_name.lower() == "auto":
                    from providers.registry import ModelProviderRegistry

                    # Use tool-specific fallback model for capacity estimation
                    # This properly handles different providers (OpenAI=200K, Gemini=1M)
                    tool_category = self.get_model_category()
                    fallback_model = ModelProviderRegistry.get_preferred_fallback_model(tool_category)
                    logger.debug(
                        f"[FILES] {self.name}: Auto mode detected, using {fallback_model} "
                        f"for {tool_category.value} tool capacity estimation"
                    )

                    try:
                        provider = self.get_model_provider(fallback_model)
                        capabilities = provider.get_capabilities(fallback_model)

                        # Calculate content allocation based on model capacity
                        if capabilities.context_window < 300_000:
                            # Smaller context models: 60% content, 40% response
                            model_content_tokens = int(capabilities.context_window * 0.6)
                        else:
                            # Larger context models: 80% content, 20% response
                            model_content_tokens = int(capabilities.context_window * 0.8)

                        effective_max_tokens = model_content_tokens - reserve_tokens
                        logger.debug(
                            f"[FILES] {self.name}: Using {fallback_model} capacity for auto mode: "
                            f"{model_content_tokens:,} content tokens from {capabilities.context_window:,} total"
                        )
                    except (ValueError, AttributeError) as e:
                        # Handle specific errors: provider not found, model not supported, missing attributes
                        logger.warning(
                            f"[FILES] {self.name}: Could not get capabilities for fallback model {fallback_model}: {type(e).__name__}: {e}"
                        )
                        # Fall back to conservative default for safety
                        effective_max_tokens = 100_000 - reserve_tokens
                    except Exception as e:
                        # Catch any other unexpected errors
                        logger.error(
                            f"[FILES] {self.name}: Unexpected error getting model capabilities: {type(e).__name__}: {e}"
                        )
                        effective_max_tokens = 100_000 - reserve_tokens
                else:
                    # Normal mode - use the specified model
                    try:
                        provider = self.get_model_provider(model_name)
                        capabilities = provider.get_capabilities(model_name)

                        # Calculate content allocation based on model capacity
                        if capabilities.context_window < 300_000:
                            # Smaller context models: 60% content, 40% response
                            model_content_tokens = int(capabilities.context_window * 0.6)
                        else:
                            # Larger context models: 80% content, 20% response
                            model_content_tokens = int(capabilities.context_window * 0.8)

                        effective_max_tokens = model_content_tokens - reserve_tokens
                        logger.debug(
                            f"[FILES] {self.name}: Using model-specific limit for {model_name}: "
                            f"{model_content_tokens:,} content tokens from {capabilities.context_window:,} total"
                        )
                    except (ValueError, AttributeError) as e:
                        # Handle specific errors: provider not found, model not supported, missing attributes
                        logger.warning(
                            f"[FILES] {self.name}: Could not get model capabilities for {model_name}: {type(e).__name__}: {e}"
                        )
                        # Fall back to conservative default for safety
                        effective_max_tokens = 100_000 - reserve_tokens
                    except Exception as e:
                        # Catch any other unexpected errors
                        logger.error(
                            f"[FILES] {self.name}: Unexpected error getting model capabilities: {type(e).__name__}: {e}"
                        )
                        effective_max_tokens = 100_000 - reserve_tokens

        # Ensure we have a reasonable minimum budget
        effective_max_tokens = max(1000, effective_max_tokens)

        files_to_embed = self.filter_new_files(request_files, continuation_id)
        logger.debug(f"[FILES] {self.name}: Will embed {len(files_to_embed)} files after filtering")

        # Log the specific files for debugging/testing
        if files_to_embed:
            logger.info(
                f"[FILE_PROCESSING] {self.name} tool will embed new files: {', '.join([os.path.basename(f) for f in files_to_embed])}"
            )
        else:
            logger.info(
                f"[FILE_PROCESSING] {self.name} tool: No new files to embed (all files already in conversation history)"
            )

        content_parts = []
        actually_processed_files = []

        # Separate images from text files
        from utils.file_types import IMAGE_EXTENSIONS

        text_files = []
        image_files = []

        for file_path in files_to_embed:
            if any(file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_files.append(file_path)
            else:
                text_files.append(file_path)

        # Store images for later use by execute method
        self._current_images = image_files
        logger.debug(f"[FILES] {self.name}: Separated {len(text_files)} text files and {len(image_files)} image files")

        # Read content of new text files only (images are handled separately)
        if text_files:
            logger.debug(f"{self.name} tool embedding {len(text_files)} new text files: {', '.join(text_files)}")
            logger.debug(
                f"[FILES] {self.name}: Starting file embedding with token budget {effective_max_tokens + reserve_tokens:,}"
            )
            try:
                # Before calling read_files, expand directories to get individual file paths
                from utils.file_utils import expand_paths

                expanded_files = expand_paths(text_files)
                logger.debug(
                    f"[FILES] {self.name}: Expanded {len(text_files)} text file paths to {len(expanded_files)} individual files"
                )

                file_content = read_files(
                    text_files,
                    max_tokens=effective_max_tokens + reserve_tokens,
                    reserve_tokens=reserve_tokens,
                    include_line_numbers=self.wants_line_numbers_by_default(),
                )
                self._validate_token_limit(file_content, context_description)
                content_parts.append(file_content)

                # Track the expanded files as actually processed
                actually_processed_files.extend(expanded_files)

                # Estimate tokens for debug logging
                from utils.token_utils import estimate_tokens

                content_tokens = estimate_tokens(file_content)
                logger.debug(
                    f"{self.name} tool successfully embedded {len(files_to_embed)} files ({content_tokens:,} tokens)"
                )
                logger.debug(f"[FILES] {self.name}: Successfully embedded files - {content_tokens:,} tokens used")
                logger.debug(
                    f"[FILES] {self.name}: Actually processed {len(actually_processed_files)} individual files"
                )
            except Exception as e:
                logger.error(f"{self.name} tool failed to embed files {files_to_embed}: {type(e).__name__}: {e}")
                logger.debug(f"[FILES] {self.name}: File embedding failed - {type(e).__name__}: {e}")
                raise
        else:
            logger.debug(f"[FILES] {self.name}: No files to embed after filtering")

        # Add note about image files if any
        if image_files:
            image_note = "\n\n--- IMAGE FILES ---\n"
            image_note += f"The following {len(image_files)} image files will be processed by the vision model:\n"
            for img in image_files:
                image_note += f"  - {img}\n"
            image_note += "--- END IMAGE FILES ---\n"
            content_parts.append(image_note)
            actually_processed_files.extend(image_files)

        # Generate note about files already in conversation history
        if continuation_id and len(files_to_embed) < len(request_files):
            embedded_files = self.get_conversation_embedded_files(continuation_id)
            skipped_files = [f for f in request_files if f in embedded_files]
            if skipped_files:
                logger.debug(
                    f"{self.name} tool skipping {len(skipped_files)} files already in conversation history: {', '.join(skipped_files)}"
                )
                logger.debug(f"[FILES] {self.name}: Adding note about {len(skipped_files)} skipped files")
                if content_parts:
                    content_parts.append("\n\n")
                note_lines = [
                    "--- NOTE: Additional files referenced in conversation history ---",
                    "The following files are already available in our conversation context:",
                    "\n".join(f"  - {f}" for f in skipped_files),
                    "--- END NOTE ---",
                ]
                content_parts.append("\n".join(note_lines))
            else:
                logger.debug(f"[FILES] {self.name}: No skipped files to note")

        # Determine file handling mode
        if file_handling_mode is None:
            # Try to get from arguments
            args_to_use = arguments or getattr(self, "_current_arguments", {})
            file_handling_mode = args_to_use.get("file_handling_mode", "embedded")

        # Handle different file modes
        file_references = None

        if file_handling_mode == "embedded":
            # Default behavior - return full content
            result = "".join(content_parts) if content_parts else ""
            logger.debug(
                f"[FILES] {self.name}: _prepare_file_content_for_prompt returning {len(result)} chars, {len(actually_processed_files)} processed files"
            )
            return result, actually_processed_files, None

        elif file_handling_mode in ["summary", "reference"]:
            # Store files and get references
            files_content = {}
            for file_path in actually_processed_files:
                # Read individual file content
                file_content, _ = read_file_content(file_path)
                if file_content:
                    files_content[file_path] = file_content

            # Store files and get references
            file_references = self._store_files_for_reference(files_content, continuation_id)

            if file_handling_mode == "summary":
                # Return summaries in the prompt for AI to analyze
                summary_parts = []
                summary_parts.append("\n=== FILE SUMMARIES ===\n")
                summary_parts.append("The following files have been stored for analysis:\n\n")

                for ref in file_references:
                    summary_parts.append(f"📄 {ref.file_path}\n")
                    summary_parts.append(f"   Summary: {ref.summary}\n")
                    summary_parts.append(f"   Reference ID: {ref.reference_id}\n")
                    summary_parts.append(f"   Size: {ref.size:,} bytes\n\n")

                summary_parts.append("=== END FILE SUMMARIES ===\n")

                # For summary mode, still include full content for AI analysis
                # but indicate that Claude will only see summaries
                full_prompt_parts = content_parts + [
                    "\n\nNOTE: The above files are provided for your analysis. "
                    + "Claude will only receive file summaries in the response, not the full content.\n"
                ]

                result = "".join(full_prompt_parts) if full_prompt_parts else ""
                logger.debug(
                    f"[FILES] {self.name}: Summary mode - stored {len(file_references)} files, returning {len(result)} chars for AI analysis"
                )
                return result, actually_processed_files, file_references

            else:  # reference mode
                # Only return reference information
                ref_parts = []
                ref_parts.append("\n=== FILE REFERENCES ===\n")
                ref_parts.append(f"The following {len(file_references)} files have been stored:\n\n")

                for ref in file_references:
                    ref_parts.append(f"- {ref.file_path} (ID: {ref.reference_id})\n")

                ref_parts.append("\n=== END FILE REFERENCES ===\n")
                ref_parts.append(
                    "\nNOTE: Full file contents are stored and can be retrieved using the reference IDs.\n"
                )

                # Still include full content for AI but with a note
                full_prompt_parts = content_parts + [
                    "\n\nNOTE: The above files are provided for your analysis. "
                    + "Claude will only receive file reference IDs in the response, not the content.\n"
                ]

                result = "".join(full_prompt_parts) if full_prompt_parts else ""
                logger.debug(f"[FILES] {self.name}: Reference mode - stored {len(file_references)} files")
                return result, actually_processed_files, file_references

        # Default fallback
        result = "".join(content_parts) if content_parts else ""
        return result, actually_processed_files, None

    def _store_file_references(self, file_references: Optional[list[FileReference]]) -> None:
        """Store file references for later use in response."""
        if file_references:
            self._current_file_references = file_references

    def _generate_file_summary(self, file_path: str, content: str) -> str:
        """
        Generate a summary for a file based on its content.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            str: Brief summary of the file
        """
        import os

        # Get file extension and name
        basename = os.path.basename(file_path)
        _, ext = os.path.splitext(basename)

        # Count lines and estimate complexity
        lines = content.split("\n")
        line_count = len(lines)

        # Determine file type
        file_type = "file"
        if ext in [".py"]:
            file_type = "Python module"
            # Count functions and classes
            import_count = sum(1 for line in lines if line.strip().startswith(("import ", "from ")))
            class_count = sum(1 for line in lines if line.strip().startswith("class "))
            func_count = sum(1 for line in lines if line.strip().startswith("def "))
            summary = f"{file_type} with {line_count} lines, {import_count} imports, {class_count} classes, {func_count} functions"
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            file_type = f"JavaScript/TypeScript {ext} file"
            summary = f"{file_type} with {line_count} lines"
        elif ext in [".java"]:
            file_type = "Java source file"
            class_count = sum(1 for line in lines if "class " in line and "{" in line)
            summary = f"{file_type} with {line_count} lines, {class_count} classes"
        elif ext in [".cpp", ".c", ".h", ".hpp"]:
            file_type = "C/C++ source file"
            summary = f"{file_type} with {line_count} lines"
        elif ext in [".go"]:
            file_type = "Go source file"
            func_count = sum(1 for line in lines if line.strip().startswith("func "))
            summary = f"{file_type} with {line_count} lines, {func_count} functions"
        elif ext in [".rs"]:
            file_type = "Rust source file"
            fn_count = sum(1 for line in lines if line.strip().startswith("fn "))
            summary = f"{file_type} with {line_count} lines, {fn_count} functions"
        elif ext in [".md"]:
            file_type = "Markdown documentation"
            # Count headers
            header_count = sum(1 for line in lines if line.strip().startswith("#"))
            summary = f"{file_type} with {line_count} lines, {header_count} sections"
        elif ext in [".json"]:
            file_type = "JSON data file"
            summary = f"{file_type} with {line_count} lines"
        elif ext in [".yaml", ".yml"]:
            file_type = "YAML configuration"
            summary = f"{file_type} with {line_count} lines"
        elif ext in [".txt"]:
            file_type = "Text file"
            word_count = len(content.split())
            summary = f"{file_type} with {line_count} lines, ~{word_count} words"
        else:
            summary = f"{basename}: {line_count} lines"

        return summary

    def _store_files_for_reference(
        self, files_content: dict[str, str], continuation_id: Optional[str] = None
    ) -> list[FileReference]:
        """
        Store files and return references.

        Args:
            files_content: Dict of file_path -> content
            continuation_id: Optional thread ID for context

        Returns:
            list[FileReference]: List of stored file references
        """
        references = []

        for file_path, content in files_content.items():
            # Generate summary
            summary = self._generate_file_summary(file_path, content)

            # Store file
            metadata = {
                "tool": self.name,
                "continuation_id": continuation_id,
            }

            file_ref = self.file_storage.store_file(
                file_path=file_path, content=content, summary=summary, metadata=metadata
            )

            references.append(file_ref)
            logger.debug(f"[FILES] Stored file {file_path} with reference {file_ref.reference_id}")

        return references

    def get_websearch_instruction(self, use_websearch: bool, tool_specific: Optional[str] = None) -> str:
        """
        Generate standardized web search instruction based on the use_websearch parameter.

        Args:
            use_websearch: Whether web search is enabled
            tool_specific: Optional tool-specific search guidance

        Returns:
            str: Web search instruction to append to prompt, or empty string
        """
        if not use_websearch:
            return ""

        base_instruction = """

WEB SEARCH CAPABILITY: You can request Claude to perform web searches to enhance your analysis with current information!

IMPORTANT: When you identify areas where web searches would significantly improve your response (such as checking current documentation, finding recent solutions, verifying best practices, or gathering community insights), you MUST explicitly instruct Claude to perform specific web searches and then respond back using the continuation_id from this response to continue the analysis.

Use clear, direct language based on the value of the search:

For valuable supplementary information: "Please perform a web search on '[specific topic/query]' and then continue this analysis using the continuation_id from this response if you find relevant information."

For important missing information: "Please search for '[specific topic/query]' and respond back with the findings using the continuation_id from this response - this information is needed to provide a complete analysis."

For critical/essential information: "SEARCH REQUIRED: Please immediately perform a web search on '[specific topic/query]' and respond back with the results using the continuation_id from this response. Cannot provide accurate analysis without this current information."

This ensures you get the most current and comprehensive information while maintaining conversation context through the continuation_id."""

        if tool_specific:
            return f"""{base_instruction}

{tool_specific}

When recommending searches, be specific about what information you need and why it would improve your analysis."""

        # Default instruction for all tools
        return f"""{base_instruction}

Consider requesting searches for:
- Current documentation and API references
- Recent best practices and patterns
- Known issues and community solutions
- Framework updates and compatibility
- Security advisories and patches
- Performance benchmarks and optimizations

When recommending searches, be specific about what information you need and why it would improve your analysis. Always remember to instruct Claude to use the continuation_id from this response when providing search results."""

    @abstractmethod
    def get_request_model(self):
        """
        Return the Pydantic model class used for validating requests.

        This model should inherit from ToolRequest and define all
        parameters specific to this tool.

        Returns:
            Type[ToolRequest]: The request model class
        """
        pass

    def validate_file_paths(self, request) -> Optional[str]:
        """
        Validate that all file paths in the request are absolute.

        This is a critical security function that prevents path traversal attacks
        and ensures all file access is properly controlled. All file paths must
        be absolute to avoid ambiguity and security issues.

        Args:
            request: The validated request object

        Returns:
            Optional[str]: Error message if validation fails, None if all paths are valid
        """
        # Check if request has 'files' attribute (used by most tools)
        if hasattr(request, "files") and request.files:
            for file_path in request.files:
                if not os.path.isabs(file_path):
                    return (
                        f"Error: All file paths must be absolute. "
                        f"Received relative path: {file_path}\n"
                        f"Please provide the full absolute path starting with '/'"
                    )

        # Check if request has 'path' attribute (used by review_changes tool)
        if hasattr(request, "path") and request.path:
            if not os.path.isabs(request.path):
                return (
                    f"Error: Path must be absolute. "
                    f"Received relative path: {request.path}\n"
                    f"Please provide the full absolute path starting with '/'"
                )

        return None

    def check_prompt_size(self, text: str) -> Optional[dict[str, Any]]:
        """
        Check if USER INPUT text is too large for MCP transport boundary.

        IMPORTANT: This method should ONLY be used to validate user input that crosses
        the Claude CLI ↔ MCP Server transport boundary. It should NOT be used to limit
        internal MCP Server operations.

        MCP Protocol Boundaries:
        Claude CLI ←→ MCP Server ←→ External Model
            ↑                              ↑
        This limit applies here      This is NOT limited

        The MCP protocol has a combined request+response limit of ~25K tokens.
        To ensure adequate space for MCP Server → Claude CLI responses, we limit
        user input to 50K characters (roughly ~10-12K tokens). Larger user prompts
        are handled by having Claude save them to prompt.txt files, bypassing MCP's
        transport constraints while preserving response capacity.

        What should be checked with this method:
        - request.prompt field (user input from Claude CLI)
        - prompt.txt file content (alternative user input)
        - Other direct user input fields

        What should NOT be checked with this method:
        - System prompts added internally
        - File content embedded by tools
        - Conversation history from Redis
        - Complete prompts sent to external models

        Args:
            text: The user input text to check (NOT internal prompt content)

        Returns:
            Optional[Dict[str, Any]]: Response asking for file handling if too large, None otherwise
        """
        if text and len(text) > MCP_PROMPT_SIZE_LIMIT:
            return {
                "status": "resend_prompt",
                "content": (
                    f"MANDATORY ACTION REQUIRED: The prompt is too large for MCP's token limits (>{MCP_PROMPT_SIZE_LIMIT:,} characters). "
                    "YOU MUST IMMEDIATELY save the prompt text to a temporary file named 'prompt.txt' in the working directory. "
                    "DO NOT attempt to shorten or modify the prompt. SAVE IT AS-IS to 'prompt.txt'. "
                    "Then resend the request with the absolute file path to 'prompt.txt' in the files parameter, "
                    "along with any other files you wish to share as context. Leave the prompt text itself empty or very brief in the new request. "
                    "This is the ONLY way to handle large prompts - you MUST follow these exact steps."
                ),
                "content_type": "text",
                "metadata": {
                    "prompt_size": len(text),
                    "limit": MCP_PROMPT_SIZE_LIMIT,
                    "instructions": "MANDATORY: Save prompt to 'prompt.txt' in current folder and include absolute path in files parameter. DO NOT modify or shorten the prompt.",
                },
            }
        return None

    def handle_prompt_file(self, files: Optional[list[str]]) -> tuple[Optional[str], Optional[list[str]]]:
        """
        Check for and handle prompt.txt in the files list.

        If prompt.txt is found, reads its content and removes it from the files list.
        This file is treated specially as the main prompt, not as an embedded file.

        This mechanism allows us to work around MCP's ~25K token limit by having
        Claude save large prompts to a file, effectively using the file transfer
        mechanism to bypass token constraints while preserving response capacity.

        Args:
            files: List of file paths (will be translated for current environment)

        Returns:
            tuple: (prompt_content, updated_files_list)
        """
        if not files:
            return None, files

        prompt_content = None
        updated_files = []

        for file_path in files:
            # Check if the filename is exactly "prompt.txt"
            # This ensures we don't match files like "myprompt.txt" or "prompt.txt.bak"
            if os.path.basename(file_path) == "prompt.txt":
                try:
                    # Read prompt.txt content and extract just the text
                    content, _ = read_file_content(file_path)
                    # Extract the content between the file markers
                    if "--- BEGIN FILE:" in content and "--- END FILE:" in content:
                        lines = content.split("\n")
                        in_content = False
                        content_lines = []
                        for line in lines:
                            if line.startswith("--- BEGIN FILE:"):
                                in_content = True
                                continue
                            elif line.startswith("--- END FILE:"):
                                break
                            elif in_content:
                                content_lines.append(line)
                        prompt_content = "\n".join(content_lines)
                    else:
                        # Fallback: if it's already raw content (from tests or direct input)
                        # and doesn't have error markers, use it directly
                        if not content.startswith("\n--- ERROR"):
                            prompt_content = content
                        else:
                            prompt_content = None
                except Exception:
                    # If we can't read the file, we'll just skip it
                    # The error will be handled elsewhere
                    pass
            else:
                # Keep the original path in the files list (will be translated later by read_files)
                updated_files.append(file_path)

        return prompt_content, updated_files if updated_files else None

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """
        Execute the tool with the provided arguments.

        This is the main entry point for tool execution. It handles:
        1. Request validation using the tool's Pydantic model
        2. File path security validation
        3. Prompt preparation
        4. Model creation and configuration
        5. Response generation and formatting
        6. Error handling and recovery

        Args:
            arguments: Dictionary of arguments from the MCP client

        Returns:
            List[TextContent]: Formatted response as MCP TextContent objects
        """
        try:
            # Store arguments for access by helper methods (like _prepare_file_content_for_prompt)
            self._current_arguments = arguments

            # Set up logger for this tool execution
            logger = logging.getLogger(f"tools.{self.name}")
            logger.info(f"🔧 {self.name} tool called with arguments: {list(arguments.keys())}")

            # Validate request using the tool's Pydantic model
            # This ensures all required fields are present and properly typed
            request_model = self.get_request_model()
            request = request_model(**arguments)
            logger.debug(f"Request validation successful for {self.name}")

            # Validate file paths for security
            # This prevents path traversal attacks and ensures proper access control
            path_error = self.validate_file_paths(request)
            if path_error:
                error_output = ToolOutput(
                    status="error",
                    content=path_error,
                    content_type="text",
                )
                return [TextContent(type="text", text=error_output.model_dump_json())]

            # Check if we have continuation_id - if so, conversation history is already embedded
            continuation_id = getattr(request, "continuation_id", None)

            if continuation_id:
                # When continuation_id is present, server.py has already injected the
                # conversation history into the appropriate field. We need to check if
                # the prompt already contains conversation history marker.
                logger.debug(f"Continuing {self.name} conversation with thread {continuation_id}")

                # Store the original arguments to detect enhanced prompts
                self._has_embedded_history = False

                # Check if conversation history is already embedded in the prompt field
                field_value = getattr(request, "prompt", "")
                field_name = "prompt"

                if "=== CONVERSATION HISTORY ===" in field_value:
                    # Conversation history is already embedded, use it directly
                    prompt = field_value
                    self._has_embedded_history = True
                    logger.debug(f"{self.name}: Using pre-embedded conversation history from {field_name}")
                else:
                    # No embedded history, prepare prompt normally
                    prompt = await self.prepare_prompt(request)
                    logger.debug(f"{self.name}: No embedded history found, prepared prompt normally")
            else:
                # New conversation, prepare prompt normally
                prompt = await self.prepare_prompt(request)

                # Add follow-up instructions for new conversations
                from server import get_follow_up_instructions

                follow_up_instructions = get_follow_up_instructions(0)  # New conversation, turn 0
                prompt = f"{prompt}\n\n{follow_up_instructions}"
                logger.debug(f"Added follow-up instructions for new {self.name} conversation")

            # Extract model configuration from request or use defaults
            model_name = getattr(request, "model", None)
            if not model_name:
                from config import DEFAULT_MODEL

                model_name = DEFAULT_MODEL

            # Check if we need Claude to select a model
            # This happens when:
            # 1. The model is explicitly "auto"
            # 2. The requested model is not available
            if self._should_require_model_selection(model_name):
                # Get suggested model based on tool category
                from providers.registry import ModelProviderRegistry

                tool_category = self.get_model_category()
                suggested_model = ModelProviderRegistry.get_preferred_fallback_model(tool_category)

                # Build error message based on why selection is required
                if model_name.lower() == "auto":
                    error_message = (
                        f"Model parameter is required in auto mode. "
                        f"Suggested model for {self.name}: '{suggested_model}' "
                        f"(category: {tool_category.value})"
                    )
                else:
                    # Model was specified but not available
                    # Get list of available models
                    available_models = self._get_available_models()

                    error_message = (
                        f"Model '{model_name}' is not available with current API keys. "
                        f"Available models: {', '.join(available_models)}. "
                        f"Suggested model for {self.name}: '{suggested_model}' "
                        f"(category: {tool_category.value})"
                    )

                error_output = ToolOutput(
                    status="error",
                    content=error_message,
                    content_type="text",
                )
                return [TextContent(type="text", text=error_output.model_dump_json())]

            # Store model name for use by helper methods like _prepare_file_content_for_prompt
            # Only set this after auto mode validation to prevent "auto" being used as a model name
            self._current_model_name = model_name

            temperature = getattr(request, "temperature", None)
            if temperature is None:
                temperature = self.get_default_temperature()
            thinking_mode = getattr(request, "thinking_mode", None)
            if thinking_mode is None:
                thinking_mode = self.get_default_thinking_mode()

            # Get the appropriate model provider
            provider = self.get_model_provider(model_name)

            # Validate and correct temperature for this model
            temperature, temp_warnings = self._validate_and_correct_temperature(model_name, temperature)

            # Log any temperature corrections
            for warning in temp_warnings:
                logger.warning(warning)

            # Get system prompt for this tool
            system_prompt = self.get_system_prompt()

            # Generate AI response using the provider
            logger.info(f"Sending request to {provider.get_provider_type().value} API for {self.name}")
            logger.info(f"Using model: {model_name} via {provider.get_provider_type().value} provider")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            # Generate content with provider abstraction
            # Add service_tier for OpenAI models if using flex processing
            generation_kwargs = {}
            if (
                provider.get_provider_type().value == "openai"
                and model_name in ["o3", "o3-mini"]
                and config.OPENAI_USE_FLEX_PROCESSING
            ):
                # Use flex service tier for OpenAI models to reduce costs
                generation_kwargs["service_tier"] = "flex"
                logger.info(f"Using Flex Processing service tier for OpenAI model {model_name}")

            # Get images if any were separated during file processing
            images = getattr(self, "_current_images", None)

            model_response = provider.generate_content(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                thinking_mode=thinking_mode if provider.supports_thinking_mode(model_name) else None,
                images=images,
                **generation_kwargs,
            )

            logger.info(f"Received response from {provider.get_provider_type().value} API for {self.name}")
            
            # Log raw response content for debugging
            if model_response.content:
                logger.debug(f"[{self.name.upper()} RESPONSE] Content length: {len(model_response.content)} chars")
                logger.debug(f"[{self.name.upper()} RESPONSE] First 500 chars: {model_response.content[:500]}...")
                if not model_response.content.strip():
                    logger.warning(f"[{self.name.upper()} RESPONSE] Response content is empty or whitespace only!")
            else:
                logger.warning(f"[{self.name.upper()} RESPONSE] No content in response object!")

            # Check if we fell back from flex tier
            if model_response.metadata.get("service_tier_fallback"):
                logger.info("Used standard service tier after Flex Processing tier was unavailable")

            # Process the model's response
            if model_response.content:
                raw_text = model_response.content

                # Parse response to check for clarification requests or format output
                # Pass model info for conversation tracking
                model_info = {"provider": provider, "model_name": model_name, "model_response": model_response}

                # Get file references if they were stored
                file_references = getattr(self, "_current_file_references", None)

                tool_output = self._parse_response(raw_text, request, model_info, file_references)
                logger.info(f"✅ {self.name} tool completed successfully")

            else:
                # Handle cases where the model couldn't generate a response
                # This might happen due to safety filters or other constraints
                finish_reason = model_response.metadata.get("finish_reason", "Unknown")
                logger.warning(f"Response blocked or incomplete for {self.name}. Finish reason: {finish_reason}")
                tool_output = ToolOutput(
                    status="error",
                    content=f"Response blocked or incomplete. Finish reason: {finish_reason}",
                    content_type="text",
                )

            # Return standardized JSON response for consistent client handling
            return [TextContent(type="text", text=tool_output.model_dump_json())]

        except Exception as e:
            # Catch all exceptions to prevent server crashes
            # Return error information in standardized format
            logger = logging.getLogger(f"tools.{self.name}")
            error_msg = str(e)

            # Check if this is an MCP size check error from prepare_prompt
            if error_msg.startswith("MCP_SIZE_CHECK:"):
                logger.info(f"MCP prompt size limit exceeded in {self.name}")
                tool_output_json = error_msg[15:]  # Remove "MCP_SIZE_CHECK:" prefix
                return [TextContent(type="text", text=tool_output_json)]

            # Check if this is a 500 INTERNAL error that asks for retry
            if "500 INTERNAL" in error_msg and "Please retry" in error_msg:
                logger.warning(f"500 INTERNAL error in {self.name} - attempting retry")
                try:
                    # Single retry attempt using provider
                    retry_response = provider.generate_content(
                        prompt=prompt,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        thinking_mode=thinking_mode if provider.supports_thinking_mode(model_name) else None,
                        images=images,
                    )

                    if retry_response.content:
                        # If successful, process normally
                        retry_model_info = {
                            "provider": provider,
                            "model_name": model_name,
                            "model_response": retry_response,
                        }
                        tool_output = self._parse_response(retry_response.content, request, retry_model_info)
                        return [TextContent(type="text", text=tool_output.model_dump_json())]

                except Exception as retry_e:
                    logger.error(f"Retry failed for {self.name} tool: {str(retry_e)}")
                    error_msg = f"Tool failed after retry: {str(retry_e)}"

            logger.error(f"Error in {self.name} tool execution: {error_msg}", exc_info=True)

            error_output = ToolOutput(
                status="error",
                content=f"Error in {self.name}: {error_msg}",
                content_type="text",
            )
            return [TextContent(type="text", text=error_output.model_dump_json())]

    def _parse_response(
        self,
        raw_text: str,
        request,
        model_info: Optional[dict] = None,
        file_references: Optional[list[FileReference]] = None,
    ) -> ToolOutput:
        """
        Parse the raw response and check for clarification requests.

        This method formats the response and always offers a continuation opportunity
        unless max conversation turns have been reached.

        Args:
            raw_text: The raw text response from the model
            request: The original request for context
            model_info: Optional dict with model metadata
            file_references: Optional list of FileReference objects for summary/reference modes

        Returns:
            ToolOutput: Standardized output object
        """
        logger = logging.getLogger(f"tools.{self.name}")

        try:
            # Try to parse as JSON to check for special status requests
            potential_json = json.loads(raw_text.strip())

            if isinstance(potential_json, dict) and "status" in potential_json:
                status_key = potential_json.get("status")
                status_model = SPECIAL_STATUS_MODELS.get(status_key)

                if status_model:
                    try:
                        # Use Pydantic for robust validation of the special status
                        parsed_status = status_model.model_validate(potential_json)
                        logger.debug(f"{self.name} tool detected special status: {status_key}")

                        # Extract model information for metadata
                        metadata = {
                            "original_request": (
                                request.model_dump() if hasattr(request, "model_dump") else str(request)
                            )
                        }
                        if model_info:
                            model_name = model_info.get("model_name")
                            if model_name:
                                metadata["model_used"] = model_name

                        return ToolOutput(
                            status=status_key,
                            content=parsed_status.model_dump_json(),
                            content_type="json",
                            metadata=metadata,
                        )

                    except Exception as e:
                        # Invalid payload for known status, log warning and continue as normal response
                        logger.warning(f"Invalid {status_key} payload: {e}")

        except (json.JSONDecodeError, ValueError, TypeError):
            # Not a JSON special status request, treat as normal response
            pass

        # Normal text response - format using tool-specific formatting
        formatted_content = self.format_response(raw_text, request, model_info)

        # Always check if we should offer Claude a continuation opportunity
        continuation_offer = self._check_continuation_opportunity(request)

        if continuation_offer:
            logger.debug(
                f"Creating continuation offer for {self.name} with {continuation_offer['remaining_turns']} turns remaining"
            )
            return self._create_continuation_offer_response(
                formatted_content, continuation_offer, request, model_info, file_references
            )
        else:
            logger.debug(f"No continuation offer created for {self.name} - max turns reached")

        # If this is a threaded conversation (has continuation_id), save the response
        continuation_id = getattr(request, "continuation_id", None)
        if continuation_id:
            request_files = getattr(request, "files", []) or []
            # Extract model metadata for conversation tracking
            model_provider = None
            model_name = None
            model_metadata = None

            if model_info:
                provider = model_info.get("provider")
                if provider:
                    model_provider = provider.get_provider_type().value
                model_name = model_info.get("model_name")
                model_response = model_info.get("model_response")
                if model_response:
                    model_metadata = {"usage": model_response.usage, "metadata": model_response.metadata}

            success = add_turn(
                continuation_id,
                "assistant",
                formatted_content,
                files=request_files,
                tool_name=self.name,
                model_provider=model_provider,
                model_name=model_name,
                model_metadata=model_metadata,
            )
            if not success:
                logging.warning(f"Failed to add turn to thread {continuation_id} for {self.name}")

        # Determine content type based on the formatted content
        content_type = (
            "markdown" if any(marker in formatted_content for marker in ["##", "**", "`", "- ", "1. "]) else "text"
        )

        # Extract model information for metadata
        metadata = {"tool_name": self.name}
        if model_info:
            model_name = model_info.get("model_name")
            if model_name:
                metadata["model_used"] = model_name

        # Add file references if in summary/reference mode
        file_refs_for_output = None
        if file_references:
            file_refs_for_output = [ref.to_dict() for ref in file_references]

        return ToolOutput(
            status="success",
            content=formatted_content,
            content_type=content_type,
            metadata=metadata,
            file_references=file_refs_for_output,
        )

    def _check_continuation_opportunity(self, request) -> Optional[dict]:
        """
        Check if we should offer Claude a continuation opportunity.

        This is called when Gemini doesn't ask a follow-up question, but we want
        to give Claude the chance to continue the conversation if needed.

        Args:
            request: The original request

        Returns:
            Dict with continuation data if opportunity should be offered, None otherwise
        """
        # Skip continuation offers in test mode
        import os

        if os.getenv("PYTEST_CURRENT_TEST"):
            return None

        continuation_id = getattr(request, "continuation_id", None)

        try:
            if continuation_id:
                # Check remaining turns in thread chain
                from utils.conversation_memory import get_thread_chain

                chain = get_thread_chain(continuation_id)
                if chain:
                    # Count total turns across all threads in chain
                    total_turns = sum(len(thread.turns) for thread in chain)
                    remaining_turns = MAX_CONVERSATION_TURNS - total_turns - 1  # -1 for this response
                else:
                    # Thread not found, don't offer continuation
                    return None
            else:
                # New conversation, we have MAX_CONVERSATION_TURNS - 1 remaining
                # (since this response will be turn 1)
                remaining_turns = MAX_CONVERSATION_TURNS - 1

            if remaining_turns <= 0:
                return None

            # Offer continuation opportunity
            return {"remaining_turns": remaining_turns, "tool_name": self.name}
        except Exception:
            # If anything fails, don't offer continuation
            return None

    def _create_continuation_offer_response(
        self,
        content: str,
        continuation_data: dict,
        request,
        model_info: Optional[dict] = None,
        file_references: Optional[list[FileReference]] = None,
    ) -> ToolOutput:
        """
        Create a response offering Claude the opportunity to continue conversation.

        Args:
            content: The main response content
            continuation_data: Dict containing remaining_turns and tool_name
            request: Original request for context
            model_info: Optional dict with model metadata
            file_references: Optional list of FileReference objects

        Returns:
            ToolOutput configured with continuation offer
        """
        try:
            # Create new thread for potential continuation (with parent link if continuing)
            continuation_id = getattr(request, "continuation_id", None)
            thread_id = create_thread(
                tool_name=self.name,
                initial_request=request.model_dump() if hasattr(request, "model_dump") else {},
                parent_thread_id=continuation_id,  # Link to parent if this is a continuation
            )

            # Add this response as the first turn (assistant turn)
            # Use actually processed files from file preparation instead of original request files
            # This ensures directories are tracked as their individual expanded files
            request_files = getattr(self, "_actually_processed_files", []) or getattr(request, "files", []) or []
            # Extract model metadata
            model_provider = None
            model_name = None
            model_metadata = None

            if model_info:
                provider = model_info.get("provider")
                if provider:
                    model_provider = provider.get_provider_type().value
                model_name = model_info.get("model_name")
                model_response = model_info.get("model_response")
                if model_response:
                    model_metadata = {"usage": model_response.usage, "metadata": model_response.metadata}

            add_turn(
                thread_id,
                "assistant",
                content,
                files=request_files,
                tool_name=self.name,
                model_provider=model_provider,
                model_name=model_name,
                model_metadata=model_metadata,
            )

            # Create continuation offer
            remaining_turns = continuation_data["remaining_turns"]
            continuation_offer = ContinuationOffer(
                continuation_id=thread_id,
                note=(
                    f"If you'd like to continue this discussion or need to provide me with further details or context, "
                    f"you can use the continuation_id '{thread_id}' with any tool and any model. "
                    f"You have {remaining_turns} more exchange(s) available in this conversation thread."
                ),
                suggested_tool_params={
                    "continuation_id": thread_id,
                    "prompt": "[Your follow-up question, additional context, or further details]",
                },
                remaining_turns=remaining_turns,
            )

            # Extract model information for metadata
            metadata = {"tool_name": self.name, "thread_id": thread_id, "remaining_turns": remaining_turns}
            if model_info:
                model_name = model_info.get("model_name")
                if model_name:
                    metadata["model_used"] = model_name

            # Add file references if in summary/reference mode
            file_refs_for_output = None
            if file_references:
                file_refs_for_output = [ref.to_dict() for ref in file_references]

            return ToolOutput(
                status="continuation_available",
                content=content,
                content_type="markdown",
                continuation_offer=continuation_offer,
                metadata=metadata,
                file_references=file_refs_for_output,
            )

        except Exception as e:
            # If threading fails, return normal response but log the error
            logger = logging.getLogger(f"tools.{self.name}")
            logger.warning(f"Conversation threading failed in {self.name}: {str(e)}")
            # Extract model information for metadata
            metadata = {"tool_name": self.name, "threading_error": str(e)}
            if model_info:
                model_name = model_info.get("model_name")
                if model_name:
                    metadata["model_used"] = model_name

            return ToolOutput(
                status="success",
                content=content,
                content_type="markdown",
                metadata=metadata,
            )

    @abstractmethod
    async def prepare_prompt(self, request) -> str:
        """
        Prepare the complete prompt for the Gemini model.

        This method should combine the system prompt with the user's request
        and any additional context (like file contents) needed for the task.

        Args:
            request: The validated request object

        Returns:
            str: Complete prompt ready for the model
        """
        pass

    def format_response(self, response: str, request, model_info: Optional[dict] = None) -> str:
        """
        Format the model's response for display.

        Override this method to add tool-specific formatting like headers,
        summaries, or structured output. Default implementation returns
        the response unchanged.

        Args:
            response: The raw response from the model
            request: The original request for context
            model_info: Optional dict with model metadata (provider, model_name, model_response)

        Returns:
            str: Formatted response
        """
        return response

    def _validate_token_limit(self, text: str, context_type: str = "Context", context_window: Optional[int] = None) -> None:
        """
        Validate token limit and raise ValueError if exceeded.

        This centralizes the token limit check that was previously duplicated
        in all prepare_prompt methods across tools.

        Args:
            text: The text to check
            context_type: Description of what's being checked (for error message)
            context_window: The model's context window size (uses model's actual limit if None)

        Raises:
            ValueError: If text exceeds context_window
        """
        # Use the model's actual context window if not specified
        if context_window is None:
            if hasattr(self, '_model_context') and self._model_context:
                # Get the actual model's context window
                context_window = self._model_context.capabilities.context_window
            else:
                # Fallback to a reasonable default for modern models
                context_window = 1_000_000  # 1M tokens for Gemini-class models
        
        within_limit, estimated_tokens = check_token_limit(text, context_window)
        if not within_limit:
            raise ValueError(
                f"{context_type} too large (~{estimated_tokens:,} tokens). Maximum is {context_window:,} tokens."
            )

    def _validate_and_correct_temperature(self, model_name: str, temperature: float) -> tuple[float, list[str]]:
        """
        Validate and correct temperature for the specified model.

        Args:
            model_name: Name of the model to validate temperature for
            temperature: Temperature value to validate

        Returns:
            Tuple of (corrected_temperature, warning_messages)
        """
        try:
            provider = self.get_model_provider(model_name)
            capabilities = provider.get_capabilities(model_name)
            constraint = capabilities.temperature_constraint

            warnings = []

            if not constraint.validate(temperature):
                corrected = constraint.get_corrected_value(temperature)
                warning = (
                    f"Temperature {temperature} invalid for {model_name}. "
                    f"{constraint.get_description()}. Using {corrected} instead."
                )
                warnings.append(warning)
                return corrected, warnings

            return temperature, warnings

        except Exception as e:
            # If validation fails for any reason, use the original temperature
            # and log a warning (but don't fail the request)
            logger = logging.getLogger(f"tools.{self.name}")
            logger.warning(f"Temperature validation failed for {model_name}: {e}")
            return temperature, [f"Temperature validation failed: {e}"]

    def get_model_provider(self, model_name: str) -> ModelProvider:
        """
        Get a model provider for the specified model.

        Args:
            model_name: Name of the model to use (can be provider-specific or generic)

        Returns:
            ModelProvider instance configured for the model

        Raises:
            ValueError: If no provider supports the requested model
        """
        # Get provider from registry
        provider = ModelProviderRegistry.get_provider_for_model(model_name)

        if not provider:
            # Try to determine provider from model name patterns
            if "gemini" in model_name.lower() or model_name.lower() in ["flash", "pro"]:
                # Register Gemini provider if not already registered
                from providers.base import ProviderType
                from providers.gemini import GeminiModelProvider

                ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
                provider = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
            elif "gpt" in model_name.lower() or "o3" in model_name.lower():
                # Register OpenAI provider if not already registered
                from providers.base import ProviderType
                from providers.openai_provider import OpenAIModelProvider

                ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
                provider = ModelProviderRegistry.get_provider(ProviderType.OPENAI)

        if not provider:
            raise ValueError(
                f"No provider found for model '{model_name}'. "
                f"Ensure the appropriate API key is set and the model name is correct."
            )

        return provider
