"""
Tool for pre-commit validation of git changes across multiple repositories.

Design Note - File Content in Multiple Sections:
Files may legitimately appear in both "Git Diffs" and "Additional Context Files" sections:
- Git Diffs: Shows changed lines + limited context (marked with "BEGIN DIFF" / "END DIFF")
- Additional Context: Shows complete file content (marked with "BEGIN FILE" / "END FILE")
This provides comprehensive context for AI analysis - not a duplication bug.
"""

import os
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from systemprompts import PRECOMMIT_PROMPT
from utils.git_utils import find_git_repositories, get_git_status, run_git_command
from utils.token_utils import estimate_tokens

from .base import BaseTool, ToolRequest

# Conservative fallback for token limits
DEFAULT_CONTEXT_WINDOW = 200_000


class PrecommitRequest(ToolRequest):
    """Request model for precommit tool"""

    path: str = Field(
        ...,
        description="Starting directory to search for git repositories (must be absolute path).",
    )
    prompt: Optional[str] = Field(
        None,
        description="The original user request description for the changes. Provides critical context for the review. If original request is limited or not available, Claude MUST study the changes carefully, think deeply about the implementation intent, analyze patterns across all modifications, infer the logic and requirements from the code changes and provide a thorough starting point.",
    )
    compare_to: Optional[str] = Field(
        None,
        description="Optional: A git ref (branch, tag, commit hash) to compare against. If not provided, reviews local staged and unstaged changes.",
    )
    include_staged: bool = Field(
        True,
        description="Include staged changes in the review. Only applies if 'compare_to' is not set.",
    )
    include_unstaged: bool = Field(
        True,
        description="Include uncommitted (unstaged) changes in the review. Only applies if 'compare_to' is not set.",
    )
    focus_on: Optional[str] = Field(
        None,
        description="Specific aspects to focus on (e.g., 'logic for user authentication', 'database query efficiency').",
    )
    review_type: Literal["full", "security", "performance", "quick"] = Field(
        "full", description="Type of review to perform on the changes."
    )
    severity_filter: Literal["critical", "high", "medium", "all"] = Field(
        "all",
        description="Minimum severity level to report on the changes.",
    )
    max_depth: int = Field(
        5,
        description="Maximum depth to search for nested git repositories to prevent excessive recursion.",
    )
    temperature: Optional[float] = Field(
        None,
        description="Temperature for the response (0.0 to 1.0). Lower values are more focused and deterministic.",
        ge=0.0,
        le=1.0,
    )
    thinking_mode: Optional[Literal["minimal", "low", "medium", "high", "max"]] = Field(
        None, description="Thinking depth mode for the assistant."
    )
    files: Optional[list[str]] = Field(
        None,
        description="Optional files or directories to provide as context (must be absolute paths). These files are not part of the changes but provide helpful context like configs, docs, or related code.",
    )


class Precommit(BaseTool):
    """Tool for pre-commit validation of git changes across multiple repositories."""

    def get_name(self) -> str:
        return "precommit"

    def get_description(self) -> str:
        return (
            "PRECOMMIT VALIDATION FOR GIT CHANGES - ALWAYS use this tool before creating any git commit! "
            "Comprehensive pre-commit validation that catches bugs, security issues, incomplete implementations, "
            "and ensures changes match the original requirements. Searches all git repositories recursively and "
            "provides deep analysis of staged/unstaged changes. Essential for code quality and preventing bugs. "
            "Use this before committing, when reviewing changes, checking your changes, validating changes, "
            "or when you're about to commit or ready to commit. Claude should proactively suggest using this tool "
            "whenever the user mentions committing or when changes are complete. "
            "When original request context is unavailable, Claude MUST think deeply about implementation intent, "
            "analyze patterns across modifications, infer business logic and requirements from code changes, "
            "and provide comprehensive insights about what was accomplished and completion status. "
            "Choose thinking_mode based on changeset size: 'low' for small focused changes, "
            "'medium' for standard commits (default), 'high' for large feature branches or complex refactoring, "
            "'max' for critical releases or when reviewing extensive changes across multiple systems. "
            "Note: If you're not currently using a top-tier model such as Opus 4 or above, these tools can provide enhanced capabilities."
        )

    def get_input_schema(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "title": "PrecommitRequest",
            "description": "Request model for precommit tool",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Starting directory to search for git repositories (must be absolute path).",
                },
                "model": self.get_model_field_schema(),
                "prompt": {
                    "type": "string",
                    "description": "The original user request description for the changes. Provides critical context for the review. If original request is limited or not available, Claude MUST study the changes carefully, think deeply about the implementation intent, analyze patterns across all modifications, infer the logic and requirements from the code changes and provide a thorough starting point.",
                },
                "compare_to": {
                    "type": "string",
                    "description": "Optional: A git ref (branch, tag, commit hash) to compare against. If not provided, reviews local staged and unstaged changes.",
                },
                "include_staged": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include staged changes in the review. Only applies if 'compare_to' is not set.",
                },
                "include_unstaged": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include uncommitted (unstaged) changes in the review. Only applies if 'compare_to' is not set.",
                },
                "focus_on": {
                    "type": "string",
                    "description": "Specific aspects to focus on (e.g., 'logic for user authentication', 'database query efficiency').",
                },
                "review_type": {
                    "type": "string",
                    "enum": ["full", "security", "performance", "quick"],
                    "default": "full",
                    "description": "Type of review to perform on the changes.",
                },
                "severity_filter": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "all"],
                    "default": "all",
                    "description": "Minimum severity level to report on the changes.",
                },
                "max_depth": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum depth to search for nested git repositories to prevent excessive recursion.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature for the response (0.0 to 1.0). Lower values are more focused and deterministic.",
                    "minimum": 0,
                    "maximum": 1,
                },
                "thinking_mode": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "max"],
                    "description": "Thinking depth mode for the assistant.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional files or directories to provide as context (must be absolute paths). These files are not part of the changes but provide helpful context like configs, docs, or related code.",
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
            "required": ["path"] + (["model"] if self.is_effective_auto_mode() else []),
        }
        return schema

    def get_system_prompt(self) -> str:
        return PRECOMMIT_PROMPT

    def get_request_model(self):
        return PrecommitRequest

    def get_default_temperature(self) -> float:
        """Use analytical temperature for code review."""
        from config import TEMPERATURE_ANALYTICAL

        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> "ToolModelCategory":
        """Precommit requires thorough analysis and reasoning"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    async def prepare_prompt(self, request: PrecommitRequest) -> str:
        """Prepare the prompt with git diff information."""
        # Check for prompt.txt in files
        prompt_content, updated_files = self.handle_prompt_file(request.files)

        # If prompt.txt was found, use it as prompt
        if prompt_content:
            request.prompt = prompt_content

        # Update request files list
        if updated_files is not None:
            request.files = updated_files

        # Check user input size at MCP transport boundary (before adding internal content)
        user_content = request.prompt if request.prompt else ""
        size_check = self.check_prompt_size(user_content)
        if size_check:
            from tools.models import ToolOutput

            raise ValueError(f"MCP_SIZE_CHECK:{ToolOutput(**size_check).model_dump_json()}")

        # Use the path directly (no translation needed anymore)
        translated_path = request.path
        translated_files = request.files if request.files else []

        # Find all git repositories
        repositories = find_git_repositories(translated_path, request.max_depth)

        if not repositories:
            return "No git repositories found in the specified path."

        # Collect all diffs directly
        all_diffs = []
        repo_summaries = []
        total_tokens = 0
        max_tokens = DEFAULT_CONTEXT_WINDOW - 50000  # Reserve tokens for prompt and response

        for repo_path in repositories:
            repo_name = os.path.basename(repo_path) or "root"

            # Get status information
            status = get_git_status(repo_path)
            changed_files = []

            # Process based on mode
            if request.compare_to:
                # Validate the ref
                is_valid_ref, err_msg = run_git_command(
                    repo_path,
                    ["rev-parse", "--verify", "--quiet", request.compare_to],
                )
                if not is_valid_ref:
                    repo_summaries.append(
                        {
                            "path": repo_path,
                            "error": f"Invalid or unknown git ref '{request.compare_to}': {err_msg}",
                            "changed_files": 0,
                        }
                    )
                    continue

                # Get list of changed files
                success, files_output = run_git_command(
                    repo_path,
                    ["diff", "--name-only", f"{request.compare_to}...HEAD"],
                )
                if success and files_output.strip():
                    changed_files = [f for f in files_output.strip().split("\n") if f]

                    # Generate per-file diffs
                    for file_path in changed_files:
                        success, diff = run_git_command(
                            repo_path,
                            [
                                "diff",
                                f"{request.compare_to}...HEAD",
                                "--",
                                file_path,
                            ],
                        )
                        if success and diff.strip():
                            # Format diff with file header
                            diff_header = (
                                f"\n--- BEGIN DIFF: {repo_name} / {file_path} (compare to {request.compare_to}) ---\n"
                            )
                            diff_footer = f"\n--- END DIFF: {repo_name} / {file_path} ---\n"
                            formatted_diff = diff_header + diff + diff_footer

                            # Check token limit
                            diff_tokens = estimate_tokens(formatted_diff)
                            if total_tokens + diff_tokens <= max_tokens:
                                all_diffs.append(formatted_diff)
                                total_tokens += diff_tokens
            else:
                # Handle staged/unstaged/untracked changes
                staged_files = []
                unstaged_files = []
                untracked_files = []

                if request.include_staged:
                    success, files_output = run_git_command(repo_path, ["diff", "--name-only", "--cached"])
                    if success and files_output.strip():
                        staged_files = [f for f in files_output.strip().split("\n") if f]

                        # Generate per-file diffs for staged changes
                        # Each diff is wrapped with clear markers to distinguish from full file content
                        for file_path in staged_files:
                            success, diff = run_git_command(repo_path, ["diff", "--cached", "--", file_path])
                            if success and diff.strip():
                                # Use "BEGIN DIFF" markers (distinct from "BEGIN FILE" markers in utils/file_utils.py)
                                # This allows AI to distinguish between diff context vs complete file content
                                diff_header = f"\n--- BEGIN DIFF: {repo_name} / {file_path} (staged) ---\n"
                                diff_footer = f"\n--- END DIFF: {repo_name} / {file_path} ---\n"
                                formatted_diff = diff_header + diff + diff_footer

                                # Check token limit
                                diff_tokens = estimate_tokens(formatted_diff)
                                if total_tokens + diff_tokens <= max_tokens:
                                    all_diffs.append(formatted_diff)
                                    total_tokens += diff_tokens

                if request.include_unstaged:
                    success, files_output = run_git_command(repo_path, ["diff", "--name-only"])
                    if success and files_output.strip():
                        unstaged_files = [f for f in files_output.strip().split("\n") if f]

                        # Generate per-file diffs for unstaged changes
                        # Same clear marker pattern as staged changes above
                        for file_path in unstaged_files:
                            success, diff = run_git_command(repo_path, ["diff", "--", file_path])
                            if success and diff.strip():
                                diff_header = f"\n--- BEGIN DIFF: {repo_name} / {file_path} (unstaged) ---\n"
                                diff_footer = f"\n--- END DIFF: {repo_name} / {file_path} ---\n"
                                formatted_diff = diff_header + diff + diff_footer

                                # Check token limit
                                diff_tokens = estimate_tokens(formatted_diff)
                                if total_tokens + diff_tokens <= max_tokens:
                                    all_diffs.append(formatted_diff)
                                    total_tokens += diff_tokens

                    # Also include untracked files when include_unstaged is True
                    # Untracked files are new files that haven't been added to git yet
                    if status["untracked_files"]:
                        untracked_files = status["untracked_files"]

                        # For untracked files, show the entire file content as a "new file" diff
                        for file_path in untracked_files:
                            file_full_path = os.path.join(repo_path, file_path)
                            if os.path.exists(file_full_path) and os.path.isfile(file_full_path):
                                try:
                                    with open(file_full_path, encoding="utf-8", errors="ignore") as f:
                                        file_content = f.read()

                                    # Format as a new file diff
                                    diff_header = (
                                        f"\n--- BEGIN DIFF: {repo_name} / {file_path} (untracked - new file) ---\n"
                                    )
                                    diff_content = f"+++ b/{file_path}\n"
                                    for _line_num, line in enumerate(file_content.splitlines(), 1):
                                        diff_content += f"+{line}\n"
                                    diff_footer = f"\n--- END DIFF: {repo_name} / {file_path} ---\n"
                                    formatted_diff = diff_header + diff_content + diff_footer

                                    # Check token limit
                                    diff_tokens = estimate_tokens(formatted_diff)
                                    if total_tokens + diff_tokens <= max_tokens:
                                        all_diffs.append(formatted_diff)
                                        total_tokens += diff_tokens
                                except Exception:
                                    # Skip files that can't be read (binary, permission issues, etc.)
                                    pass

                # Combine unique files
                changed_files = list(set(staged_files + unstaged_files + untracked_files))

            # Add repository summary
            if changed_files:
                repo_summaries.append(
                    {
                        "path": repo_path,
                        "branch": status["branch"],
                        "ahead": status["ahead"],
                        "behind": status["behind"],
                        "changed_files": len(changed_files),
                        "files": changed_files[:20],  # First 20 for summary
                    }
                )

        if not all_diffs:
            return "No pending changes found in any of the git repositories."

        # Process context files if provided using standardized file reading
        context_files_content = []
        context_files_summary = []
        context_tokens = 0

        if translated_files:
            remaining_tokens = max_tokens - total_tokens

            # Use centralized file handling with filtering for duplicate prevention
            file_content, processed_files, file_references = self._prepare_file_content_for_prompt(
                translated_files,
                request.continuation_id,
                "Context files",
                max_tokens=remaining_tokens + 1000,  # Add back the reserve that was calculated
                reserve_tokens=1000,  # Small reserve for formatting
            )

            # Store file references for response formatting
            if file_references:
                self._store_file_references(file_references)
            self._actually_processed_files = processed_files

            if file_content:
                context_tokens = estimate_tokens(file_content)
                context_files_content = [file_content]
                context_files_summary.append(f"✅ Included: {len(translated_files)} context files")
            else:
                context_files_summary.append("WARNING: No context files could be read or files too large")

            total_tokens += context_tokens

        # Build the final prompt
        prompt_parts = []

        # Add original request context if provided
        if request.prompt:
            prompt_parts.append(f"## Original Request\n\n{request.prompt}\n")

        # Add review parameters
        prompt_parts.append("## Review Parameters\n")
        prompt_parts.append(f"- Review Type: {request.review_type}")
        prompt_parts.append(f"- Severity Filter: {request.severity_filter}")

        if request.focus_on:
            prompt_parts.append(f"- Focus Areas: {request.focus_on}")

        if request.compare_to:
            prompt_parts.append(f"- Comparing Against: {request.compare_to}")
        else:
            review_scope = []
            if request.include_staged:
                review_scope.append("staged")
            if request.include_unstaged:
                review_scope.append("unstaged")
            prompt_parts.append(f"- Reviewing: {' and '.join(review_scope)} changes")

        # Add repository summary
        prompt_parts.append("\n## Repository Changes Summary\n")
        prompt_parts.append(f"Found {len(repo_summaries)} repositories with changes:\n")

        for idx, summary in enumerate(repo_summaries, 1):
            prompt_parts.append(f"\n### Repository {idx}: {summary['path']}")
            if "error" in summary:
                prompt_parts.append(f"ERROR: {summary['error']}")
            else:
                prompt_parts.append(f"- Branch: {summary['branch']}")
                if summary["ahead"] or summary["behind"]:
                    prompt_parts.append(f"- Ahead: {summary['ahead']}, Behind: {summary['behind']}")
                prompt_parts.append(f"- Changed Files: {summary['changed_files']}")

                if summary["files"]:
                    prompt_parts.append("\nChanged files:")
                    for file in summary["files"]:
                        prompt_parts.append(f"  - {file}")
                    if summary["changed_files"] > len(summary["files"]):
                        prompt_parts.append(f"  ... and {summary['changed_files'] - len(summary['files'])} more files")

        # Add context files summary if provided
        if context_files_summary:
            prompt_parts.append("\n## Context Files Summary\n")
            for summary_item in context_files_summary:
                prompt_parts.append(f"- {summary_item}")

        # Add token usage summary
        if total_tokens > 0:
            prompt_parts.append(f"\nTotal context tokens used: ~{total_tokens:,}")

        # Add the diff contents with clear section markers
        # Each diff is wrapped with "--- BEGIN DIFF: ... ---" and "--- END DIFF: ... ---"
        prompt_parts.append("\n## Git Diffs\n")
        if all_diffs:
            prompt_parts.extend(all_diffs)
        else:
            prompt_parts.append("--- NO DIFFS FOUND ---")

        # Add context files content if provided
        # IMPORTANT: Files may legitimately appear in BOTH sections:
        # - Git Diffs: Show only changed lines + limited context (what changed)
        # - Additional Context: Show complete file content (full understanding)
        # This is intentional design for comprehensive AI analysis, not duplication bug.
        # Each file in this section is wrapped with "--- BEGIN FILE: ... ---" and "--- END FILE: ... ---"
        if context_files_content:
            prompt_parts.append("\n## Additional Context Files")
            prompt_parts.append(
                "The following files are provided for additional context. They have NOT been modified.\n"
            )
            prompt_parts.extend(context_files_content)

        # Add web search instruction if enabled
        websearch_instruction = self.get_websearch_instruction(
            request.use_websearch,
            """When validating changes, consider if searches for these would help:
- Best practices for new features or patterns introduced
- Security implications of the changes
- Known issues with libraries or APIs being used
- Migration guides if updating dependencies
- Performance considerations for the implemented approach""",
        )

        # Add review instructions
        prompt_parts.append("\n## Review Instructions\n")
        prompt_parts.append(
            "Please review these changes according to the system prompt guidelines. "
            "Pay special attention to alignment with the original request, completeness of implementation, "
            "potential bugs, security issues, and any edge cases not covered."
        )

        # Add instruction for requesting files if needed
        if not translated_files:
            prompt_parts.append(
                "\nIf you need additional context files to properly review these changes "
                "(such as configuration files, documentation, or related code), "
                "you may request them using the standardized JSON response format."
            )

        # Combine with system prompt and websearch instruction
        full_prompt = f"{self.get_system_prompt()}{websearch_instruction}\n\n" + "\n".join(prompt_parts)

        return full_prompt

    def format_response(self, response: str, request: PrecommitRequest, model_info: Optional[dict] = None) -> str:
        """Format the response with commit guidance"""
        return f"{response}\n\n---\n\n**Commit Status:** If no critical issues found, changes are ready for commit. Otherwise, address issues first and re-run review. Check with user before proceeding with any commit."
