"""
ThinkDeep tool - Extended reasoning and problem-solving
"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_CREATIVE
from systemprompts import THINKDEEP_PROMPT

from .base import BaseTool, ToolRequest


class ThinkDeepRequest(ToolRequest):
    """Request model for thinkdeep tool"""

    prompt: str = Field(
        ...,
        description="Your current thinking/analysis to extend and validate. IMPORTANT: Before using this tool, Claude MUST first think hard and establish a deep understanding of the topic and question by thinking through all relevant details, context, constraints, and implications. Share these extended thoughts and ideas in the prompt so the model has comprehensive information to work with for the best analysis.",
    )
    problem_context: Optional[str] = Field(
        None, description="Additional context about the problem or goal. Be as expressive as possible."
    )
    focus_areas: Optional[list[str]] = Field(
        None,
        description="Specific aspects to focus on (architecture, performance, security, etc.)",
    )
    files: Optional[list[str]] = Field(
        None,
        description="Optional file paths or directories for additional context (must be absolute paths)",
    )


class ThinkDeepTool(BaseTool):
    """Extended thinking and reasoning tool"""

    def get_name(self) -> str:
        return "thinkdeep"

    def get_description(self) -> str:
        return (
            "EXTENDED THINKING & REASONING - Your deep thinking partner for complex problems. "
            "Use this when you need to think deeper about a problem, extend your analysis, explore alternatives, or validate approaches. "
            "Perfect for: architecture decisions, complex bugs, performance challenges, security analysis. "
            "I'll challenge assumptions, find edge cases, and provide alternative solutions. "
            "IMPORTANT: Choose the appropriate thinking_mode based on task complexity - "
            "'low' for quick analysis, 'medium' for standard problems, 'high' for complex issues (default), "
            "'max' for extremely complex challenges requiring deepest analysis. "
            "When in doubt, err on the side of a higher mode for truly deep thought and evaluation. "
            "Note: If you're not currently using a top-tier model such as Opus 4 or above, these tools can provide enhanced capabilities."
        )

    def get_input_schema(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Your current thinking/analysis to extend and validate. IMPORTANT: Before using this tool, Claude MUST first think deeply and establish a deep understanding of the topic and question by thinking through all relevant details, context, constraints, and implications. Share these extended thoughts and ideas in the prompt so the model has comprehensive information to work with for the best analysis.",
                },
                "model": self.get_model_field_schema(),
                "problem_context": {
                    "type": "string",
                    "description": "Additional context about the problem or goal. Be as expressive as possible.",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific aspects to focus on (architecture, performance, security, etc.)",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional file paths or directories for additional context (must be absolute paths)",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature for creative thinking (0-1, default 0.7)",
                    "minimum": 0,
                    "maximum": 1,
                },
                "thinking_mode": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "max"],
                    "description": f"Thinking depth: minimal (0.5% of model max), low (8%), medium (33%), high (67%), max (100% of model max). Defaults to '{self.get_default_thinking_mode()}' if not specified.",
                },
                "use_websearch": {
                    "type": "boolean",
                    "description": "Enable web search for documentation, best practices, and current information. Particularly useful for: brainstorming sessions, architectural design discussions, exploring industry best practices, working with specific frameworks/technologies, researching solutions to complex problems, or when current documentation and community insights would enhance the analysis.",
                    "default": True,
                },
                "file_handling_mode": {
                    "type": "string",
                    "enum": ["embedded", "summary", "reference"],
                    "default": "embedded",
                    "description": "How to handle file content in responses. 'embedded' includes full content (default), 'summary' returns only summaries to save tokens, 'reference' stores files and returns IDs.",
                },
                "continuation_id": {
                    "type": "string",
                    "description": "Thread continuation ID for multi-turn conversations. Can be used to continue conversations across different tools. Only provide this if continuing a previous conversation thread.",
                },
            },
            "required": ["prompt"] + (["model"] if self.is_effective_auto_mode() else []),
        }

        return schema

    def get_system_prompt(self) -> str:
        return THINKDEEP_PROMPT

    def get_default_temperature(self) -> float:
        return TEMPERATURE_CREATIVE

    def get_default_thinking_mode(self) -> str:
        """ThinkDeep uses configurable thinking mode, defaults to high"""
        from config import DEFAULT_THINKING_MODE_THINKDEEP

        return DEFAULT_THINKING_MODE_THINKDEEP

    def get_model_category(self) -> "ToolModelCategory":
        """ThinkDeep requires extended reasoning capabilities"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_request_model(self):
        return ThinkDeepRequest

    async def prepare_prompt(self, request: ThinkDeepRequest) -> str:
        """Prepare the full prompt for extended thinking"""
        # Check for prompt.txt in files
        prompt_content, updated_files = self.handle_prompt_file(request.files)

        # Use prompt.txt content if available, otherwise use the prompt field
        current_analysis = prompt_content if prompt_content else request.prompt

        # Check user input size at MCP transport boundary (before adding internal content)
        size_check = self.check_prompt_size(current_analysis)
        if size_check:
            from tools.models import ToolOutput

            raise ValueError(f"MCP_SIZE_CHECK:{ToolOutput(**size_check).model_dump_json()}")

        # Update request files list
        if updated_files is not None:
            request.files = updated_files

        # Build context parts
        context_parts = [f"=== CLAUDE'S CURRENT ANALYSIS ===\n{current_analysis}\n=== END ANALYSIS ==="]

        if request.problem_context:
            context_parts.append(f"\n=== PROBLEM CONTEXT ===\n{request.problem_context}\n=== END CONTEXT ===")

        # Add reference files if provided
        if request.files:
            # Use centralized file processing logic
            continuation_id = getattr(request, "continuation_id", None)
            file_content, processed_files, file_references = self._prepare_file_content_for_prompt(
                request.files, continuation_id, "Reference files"
            )

            # Store file references for response formatting
            if file_references:
                self._store_file_references(file_references)
            self._actually_processed_files = processed_files

            if file_content:
                context_parts.append(f"\n=== REFERENCE FILES ===\n{file_content}\n=== END FILES ===")

        full_context = "\n".join(context_parts)

        # Check token limits
        self._validate_token_limit(full_context, "Context")

        # Add focus areas instruction if specified
        focus_instruction = ""
        if request.focus_areas:
            areas = ", ".join(request.focus_areas)
            focus_instruction = f"\n\nFOCUS AREAS: Please pay special attention to {areas} aspects."

        # Add web search instruction if enabled
        websearch_instruction = self.get_websearch_instruction(
            request.use_websearch,
            """When analyzing complex problems, consider if searches for these would help:
- Current documentation for specific technologies, frameworks, or APIs mentioned
- Known issues, workarounds, or community solutions for similar problems
- Recent updates, deprecations, or best practices that might affect the approach
- Official sources to verify assumptions or clarify technical details""",
        )

        # Combine system prompt with context
        full_prompt = f"""{self.get_system_prompt()}{focus_instruction}{websearch_instruction}

{full_context}

Please provide deep analysis that extends Claude's thinking with:
1. Alternative approaches and solutions
2. Edge cases and potential failure modes
3. Critical evaluation of assumptions
4. Concrete implementation suggestions
5. Risk assessment and mitigation strategies"""

        return full_prompt

    def format_response(self, response: str, request: ThinkDeepRequest, model_info: Optional[dict] = None) -> str:
        """Format the response with clear attribution and critical thinking prompt"""
        # Get the friendly model name
        model_name = "your fellow developer"
        if model_info and model_info.get("model_response"):
            model_name = model_info["model_response"].friendly_name or "your fellow developer"

        return f"""{response}

---

## Critical Evaluation Required

Claude, please critically evaluate {model_name}'s analysis by thinking hard about the following:

1. **Technical merit** - Which suggestions are valuable vs. have limitations?
2. **Constraints** - Fit with codebase patterns, performance, security, architecture
3. **Risks** - Hidden complexities, edge cases, potential failure modes
4. **Final recommendation** - Synthesize both perspectives, then ultrathink on your own to explore additional
considerations and arrive at the best technical solution. Feel free to use zen's chat tool for a follow-up discussion
if needed.

Remember: Use {model_name}'s insights to enhance, not replace, your analysis."""
