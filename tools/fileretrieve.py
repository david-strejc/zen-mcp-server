"""
File retrieval tool for accessing stored file content.

This tool allows Claude to retrieve files that were stored using the
summary or reference file handling modes.
"""

import logging
from typing import Any, Optional

from pydantic import Field

from tools.base import BaseTool, ToolRequest
from utils.file_storage import FileStorage

logger = logging.getLogger(__name__)


class FileRetrieveRequest(ToolRequest):
    """Request model for file retrieval"""

    reference_id: str = Field(..., description="The file reference ID to retrieve")


class FileRetrieveTool(BaseTool):
    """Tool for retrieving stored file content by reference ID"""

    def __init__(self):
        super().__init__()
        self.file_storage = FileStorage()

    def get_name(self) -> str:
        return "fileretrieve"

    def get_description(self) -> str:
        return (
            "FILERETRIEVE - Get stored file content by reference ID."
        )

    def get_input_schema(self) -> dict[str, Any]:
        return FileRetrieveRequest.model_json_schema()

    def get_system_prompt(self) -> str:
        return """You are a file retrieval assistant. Your role is to:
1. Retrieve and display the requested file content
2. Provide the file path and metadata
3. Format the content appropriately based on file type

Be clear and concise in your responses."""

    def get_request_model(self):
        return FileRetrieveRequest

    def get_default_temperature(self) -> float:
        return 0.1  # Very low temperature for consistent retrieval

    async def prepare_prompt(self, request: FileRetrieveRequest) -> str:
        """Prepare prompt for file retrieval"""

        # Retrieve the file
        result = self.file_storage.retrieve_file(request.reference_id)

        if not result:
            return f"""The file with reference ID '{request.reference_id}' was not found.

This could mean:
1. The file reference has expired (files are stored for 24 hours)
2. The reference ID is incorrect
3. The file was never stored

Please check the reference ID and try again."""

        content, file_ref = result

        # Format the prompt
        prompt = f"""Retrieved file content:

=== FILE INFORMATION ===
Path: {file_ref.file_path}
Reference ID: {file_ref.reference_id}
Size: {file_ref.size:,} bytes
Summary: {file_ref.summary}
Created: {file_ref.created_at}

=== FILE CONTENT ===
{content}
=== END FILE CONTENT ===

Please provide the file content to the user in a clear and formatted manner."""

        return prompt

    def format_response(self, response: str, request: FileRetrieveRequest, model_info: Optional[dict] = None) -> str:
        """Format the response - usually just pass through"""
        return response
