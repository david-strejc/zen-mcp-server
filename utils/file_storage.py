"""
File storage mechanism for MCP server.
Stores file contents in Redis with references, allowing tools to process files
without exposing full content to the initiating AI (Claude).
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Optional

from .redis_manager import get_redis_client


class FileReference:
    """Represents a stored file reference."""

    def __init__(
        self,
        file_path: str,
        reference_id: str,
        size: int,
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.file_path = file_path
        self.reference_id = reference_id
        self.size = size
        self.summary = summary or f"File: {os.path.basename(file_path)} ({size} bytes)"
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "reference_id": self.reference_id,
            "size": self.size,
            "summary": self.summary,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileReference":
        """Create FileReference from dictionary."""
        ref = cls(
            file_path=data["file_path"],
            reference_id=data["reference_id"],
            size=data["size"],
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
        )
        ref.created_at = data.get("created_at", datetime.utcnow().isoformat())
        return ref


class FileStorage:
    """Manages file storage and retrieval using Redis."""

    def __init__(self, ttl_hours: int = 24):
        """
        Initialize file storage.

        Args:
            ttl_hours: Time-to-live for stored files in hours
        """
        self.redis_client = get_redis_client()
        self.ttl = timedelta(hours=ttl_hours)
        self.file_prefix = "mcp:file:"
        self.ref_prefix = "mcp:fileref:"

    def generate_reference_id(self, file_path: str, content: str) -> str:
        """Generate a unique reference ID for a file."""
        # Use file path and content hash to generate consistent IDs
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        return f"file_{path_hash}_{content_hash}"

    def store_file(
        self, file_path: str, content: str, summary: Optional[str] = None, metadata: Optional[dict] = None
    ) -> FileReference:
        """
        Store a file and return a reference.

        Args:
            file_path: Path to the file
            content: File content
            summary: Optional summary of the file
            metadata: Optional metadata about the file

        Returns:
            FileReference object
        """
        reference_id = self.generate_reference_id(file_path, content)

        # Store file content
        content_key = f"{self.file_prefix}{reference_id}"
        self.redis_client.setex(content_key, self.ttl, content)

        # Create and store reference
        file_ref = FileReference(
            file_path=file_path, reference_id=reference_id, size=len(content), summary=summary, metadata=metadata
        )

        ref_key = f"{self.ref_prefix}{reference_id}"
        self.redis_client.setex(ref_key, self.ttl, json.dumps(file_ref.to_dict()))

        return file_ref

    def retrieve_file(self, reference_id: str) -> Optional[tuple[str, FileReference]]:
        """
        Retrieve file content and reference by ID.

        Args:
            reference_id: The file reference ID

        Returns:
            Tuple of (content, FileReference) or None if not found
        """
        content_key = f"{self.file_prefix}{reference_id}"
        ref_key = f"{self.ref_prefix}{reference_id}"

        content = self.redis_client.get(content_key)
        ref_data = self.redis_client.get(ref_key)

        if not content or not ref_data:
            return None

        file_ref = FileReference.from_dict(json.loads(ref_data))
        return content, file_ref

    def get_reference(self, reference_id: str) -> Optional[FileReference]:
        """Get file reference without content."""
        ref_key = f"{self.ref_prefix}{reference_id}"
        ref_data = self.redis_client.get(ref_key)

        if not ref_data:
            return None

        return FileReference.from_dict(json.loads(ref_data))

    def list_references(self, pattern: str = "*") -> list[FileReference]:
        """List all stored file references matching pattern."""
        references = []

        # Find all reference keys
        ref_pattern = f"{self.ref_prefix}{pattern}"
        for key in self.redis_client.scan_iter(match=ref_pattern):
            ref_data = self.redis_client.get(key)
            if ref_data:
                references.append(FileReference.from_dict(json.loads(ref_data)))

        return references

    def delete_file(self, reference_id: str) -> bool:
        """Delete a stored file and its reference."""
        content_key = f"{self.file_prefix}{reference_id}"
        ref_key = f"{self.ref_prefix}{reference_id}"

        deleted = 0
        deleted += self.redis_client.delete(content_key)
        deleted += self.redis_client.delete(ref_key)

        return deleted > 0

    def cleanup_expired(self) -> int:
        """Clean up expired files. Returns count of deleted files."""
        # Redis handles TTL automatically, this is for manual cleanup if needed
        deleted = 0

        # Check all file references
        for key in self.redis_client.scan_iter(match=f"{self.ref_prefix}*"):
            if not self.redis_client.ttl(key):
                # TTL expired or not set
                ref_id = key.decode().replace(self.ref_prefix, "")
                if self.delete_file(ref_id):
                    deleted += 1

        return deleted
