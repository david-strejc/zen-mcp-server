# Conversation Continuation Mechanism in MCP Server

## Overview

The MCP server implements a sophisticated conversation continuation mechanism that enables multi-turn conversations between Claude and AI models (Gemini, OpenAI, etc.) across different tools. The system is stateless by design, with conversation state persisted in Redis.

## Key Components

### 1. Thread Management (`utils/conversation_memory.py`)

The conversation memory module provides the core infrastructure:

- **Thread Creation**: Each conversation starts with a unique UUID thread ID
- **Turn Storage**: Conversation turns are stored with role, content, files, tool name, and model metadata
- **Thread Chaining**: Supports parent-child thread relationships for conversation chains
- **Redis Persistence**: All conversation data stored in Redis with configurable TTL (default 3 hours)

### 2. File Deduplication System

The system prevents duplicate file embeddings across conversation turns:

#### File Tracking
- Each conversation turn tracks which files were used (`turn.files`)
- `get_conversation_file_list()` extracts all unique files across all turns
- Files are embedded ONCE at the start of conversation history

#### Tool-Level Deduplication (`tools/base.py`)
- `get_conversation_embedded_files()`: Returns files already in conversation history
- `filter_new_files()`: Filters out already-embedded files
- `_prepare_file_content_for_prompt()`: Only reads and embeds NEW files

### 3. Conversation History Building

When a tool receives a `continuation_id`:

1. **History Retrieval**: `get_thread()` fetches the conversation from Redis
2. **Chain Traversal**: `get_thread_chain()` follows parent links to get full history
3. **File Collection**: All unique files from all turns are collected
4. **File Embedding**: Files are read and embedded ONCE at the start
5. **Turn Formatting**: Each turn shows role, tool used, files referenced, and content
6. **Token Management**: Respects model-specific token limits for files and history

### 4. Server Integration (`server.py`)

The server handles continuation at the transport layer:

```python
# When continuation_id is present:
1. Fetch conversation from Redis
2. Add user's new input as a turn
3. Build complete conversation history with files
4. Inject history into tool's prompt field
5. Calculate remaining token budget
6. Pass enhanced request to tool
```

### 5. Cross-Tool Continuation

Any tool can continue a conversation started by another tool:

- Tool A creates thread → returns continuation_id
- Tool B uses continuation_id → sees full history from Tool A
- Files from Tool A are available in conversation context
- Tool B only embeds NEW files not already in history

## File Handling Flow

### New Conversation
```
1. Tool receives files: [file1.py, file2.py]
2. All files are NEW → embed all files
3. Create thread, store files in turn metadata
4. Return continuation_id for future use
```

### Continued Conversation
```
1. Tool receives continuation_id + files: [file1.py, file3.py]
2. Check embedded files in history: [file1.py, file2.py]
3. Filter to new files only: [file3.py]
4. Embed only file3.py
5. Add note about files already in history
6. Update thread with new turn
```

### Directory Handling
- Directories are expanded to individual files
- Each file is tracked separately in conversation history
- Prevents re-reading same files even if directory is re-specified

## Token Budget Management

The system uses model-specific token allocation:

1. **Total Context Window**: Model's maximum tokens
2. **Response Allocation**: Reserved for model's response
3. **Content Allocation**: Available for files + history
4. **File Token Budget**: Portion for file embeddings
5. **History Token Budget**: Portion for conversation turns

Example for 1M token model:
- Total: 1,000,000 tokens
- Response: 200,000 tokens (20%)
- Content: 800,000 tokens (80%)
- Files: 400,000 tokens (50% of content)
- History: 400,000 tokens (50% of content)

## Implementation Details

### Thread Context Structure
```python
class ThreadContext:
    thread_id: str                    # UUID
    parent_thread_id: Optional[str]   # For chains
    created_at: str                   # ISO timestamp
    last_updated_at: str              # ISO timestamp
    tool_name: str                    # Original tool
    turns: list[ConversationTurn]     # All turns
    initial_context: dict             # Original request
```

### Conversation Turn Structure
```python
class ConversationTurn:
    role: str                         # "user" or "assistant"
    content: str                      # Message content
    timestamp: str                    # ISO timestamp
    files: Optional[list[str]]        # Files in this turn
    tool_name: Optional[str]          # Tool that created turn
    model_provider: Optional[str]     # Provider (google, openai)
    model_name: Optional[str]         # Specific model
    model_metadata: Optional[dict]    # Additional metadata
```

### File Deduplication Log Pattern
```
[FILE_PROCESSING] analyze tool will embed new files: config.py, utils.py
[FILE_PROCESSING] analyze tool: No new files to embed (all files already in conversation history)
[FILES] analyze: Filtering 3 requested files
[FILES] analyze: Found 2 embedded files in conversation
[FILES] analyze: After filtering: 1 new files, 2 already embedded
```

## Configuration

### Environment Variables
- `MAX_CONVERSATION_TURNS`: Maximum turns per thread (default: 20)
- `CONVERSATION_TIMEOUT_HOURS`: Redis TTL in hours (default: 3)
- `REDIS_URL`: Redis connection URL (default: redis://localhost:6379/0)

### Turn Limits
- Prevents runaway conversations
- Counts turns across entire thread chain
- Returns error when limit reached

## Best Practices

1. **Always use absolute file paths** - Required for security
2. **Track expanded files** - Directories expand to individual files
3. **Check continuation_id validity** - Handle expired threads gracefully
4. **Monitor token usage** - Respect model-specific limits
5. **Log file operations** - Use [FILE_PROCESSING] and [FILES] prefixes

## Testing

The system includes comprehensive tests:
- `test_cross_tool_comprehensive.py`: Full cross-tool workflow
- `test_cross_tool_continuation.py`: Continuation scenarios
- `test_per_tool_deduplication.py`: File deduplication logic
- `test_conversation_chain_validation.py`: Thread chaining

## Security Considerations

1. **UUID Validation**: All thread IDs validated as proper UUIDs
2. **Path Security**: Only absolute paths allowed
3. **Redis Isolation**: Each thread isolated by UUID
4. **TTL Expiration**: Automatic cleanup of old conversations
5. **Error Handling**: No sensitive information in errors