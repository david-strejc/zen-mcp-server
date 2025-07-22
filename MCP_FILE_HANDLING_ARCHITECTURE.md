# MCP Server File Handling Architecture Analysis

## Overview

This document analyzes the current file handling architecture in the Zen MCP Server to understand how we could modify it to send files to consulting AI models without including them in Claude's context.

## Current Architecture

### 1. Request Flow

```
Claude (MCP Client) → MCP Server → Tool → AI Provider (Gemini/OpenAI/etc)
                                         ↓
Claude (MCP Client) ← MCP Server ← Tool ←
```

### 2. File Handling Process

#### Current Implementation:

1. **Client Request**: Claude sends a tool request with file paths
2. **Path Validation**: `BaseTool.validate_file_paths()` ensures absolute paths
3. **File Reading**: `BaseTool._prepare_file_content_for_prompt()` reads files:
   - Calls `read_files()` to get file content
   - Adds line numbers if requested
   - Manages token budget
   - Returns formatted content with file markers
4. **Prompt Construction**: Files are embedded directly into the prompt sent to AI
5. **Response**: AI response is returned to Claude via MCP protocol

#### Key Components:

- **BaseTool**: Base class handling file validation and preparation
- **file_utils.py**: Core file reading and security functions
- **ModelProvider**: Interface for communicating with AI models
- **ToolOutput**: Standardized response format

### 3. File Content Format

Files are currently embedded with clear markers:
```
--- BEGIN FILE: /path/to/file.py ---
[file content with optional line numbers]
--- END FILE: /path/to/file.py ---
```

## Architectural Modification Options

### Option 1: File Reference System

**Concept**: Instead of embedding file content in responses, store files externally and return references.

**Implementation**:
1. Create a file storage service (using Redis or filesystem)
2. When AI needs files, store them with unique IDs
3. Return file references instead of content
4. Claude would need a separate mechanism to retrieve files

**Pros**:
- Complete separation of file content from Claude's context
- Scalable for large files
- Can implement access control

**Cons**:
- Requires Claude to support file reference retrieval
- Adds complexity to the protocol
- May not be compatible with current MCP spec

### Option 2: Streaming Response with Metadata

**Concept**: Use MCP's response metadata to indicate files were processed without including content.

**Implementation**:
1. Modify `ToolOutput` to include a `processed_files` metadata field
2. AI tools process files normally
3. Return only analysis results, not file content
4. Include file metadata (paths, sizes, summaries) in response

**Pros**:
- Works within current MCP protocol
- Claude knows which files were analyzed
- No protocol changes needed

**Cons**:
- Limited visibility into what was actually processed
- Harder to verify AI's analysis

### Option 3: Dual-Mode Operation

**Concept**: Support both embedded and reference modes based on configuration or request parameters.

**Implementation**:
1. Add `file_handling_mode` parameter to tool requests
2. In reference mode:
   - Store files in Redis with TTL
   - Return file IDs and metadata
   - Provide a separate `retrieve_file` tool
3. In embedded mode (current):
   - Continue current behavior

**Pros**:
- Backward compatible
- Flexible based on use case
- Can gradually migrate

**Cons**:
- More complex codebase
- Requires client awareness of modes

### Option 4: File Summary System

**Concept**: Instead of full file content, return AI-generated summaries.

**Implementation**:
1. AI processes files normally
2. Generate structured summaries of each file
3. Return summaries instead of content
4. Include key excerpts only when necessary

**Pros**:
- Reduces token usage
- Maintains context awareness
- Works with current protocol

**Cons**:
- Loss of detail
- Dependent on AI summary quality

## Recommended Approach: Hybrid Summary + Reference System

Based on the analysis, I recommend a hybrid approach combining Options 2, 3, and 4:

### 1. Modify ToolOutput Structure

```python
class FileReference(BaseModel):
    """Reference to a processed file"""
    path: str
    size: int
    lines: int
    summary: str
    key_sections: list[str]  # Important excerpts
    reference_id: Optional[str]  # For full retrieval if needed

class ToolOutput(BaseModel):
    # ... existing fields ...
    processed_files: Optional[list[FileReference]] = None
    file_handling_mode: Literal["embedded", "reference", "summary"] = "embedded"
```

### 2. Implement File Processing Pipeline

```python
def process_files_for_ai(files: list[str], mode: str) -> tuple[str, list[FileReference]]:
    if mode == "embedded":
        # Current behavior
        return read_files(files), []
    elif mode == "summary":
        # Process files, generate summaries
        # Return summaries in prompt, references in metadata
    elif mode == "reference":
        # Store files, return minimal prompt with IDs
```

### 3. Add Configuration Support

```python
# In BaseTool
def get_file_handling_mode(self) -> str:
    """Determine file handling mode from request or config"""
    # Check request parameter
    # Fall back to environment variable
    # Default to "embedded" for compatibility
```

### 4. Implement File Retrieval Tool

```python
class RetrieveFileTool(BaseTool):
    """Tool to retrieve previously processed files by reference"""
    def execute(self, reference_id: str) -> str:
        # Retrieve from Redis or filesystem
        # Return file content
```

## Implementation Considerations

### 1. Storage Backend
- Redis: Fast, TTL support, already in use
- Filesystem: Simple, persistent, no size limits
- S3/Object Storage: Scalable, but adds dependency

### 2. Security
- Maintain current path validation
- Add access control for references
- Implement reference expiration

### 3. Backward Compatibility
- Default to embedded mode
- Gradual migration path
- Clear documentation

### 4. Performance
- Lazy loading of file content
- Efficient summary generation
- Caching strategies

## Next Steps

1. Prototype the FileReference model
2. Implement summary generation in BaseTool
3. Add Redis-based file storage
4. Create retrieve_file tool
5. Test with various file sizes and types
6. Document new modes for users

## Conclusion

The recommended hybrid approach provides flexibility while maintaining compatibility. It allows tools to process files without necessarily including full content in Claude's context, while still providing enough information for Claude to understand what was analyzed. The system can be implemented incrementally without breaking existing functionality.