"""
Redis manager for centralized Redis client management.
"""

from utils.conversation_memory import get_redis_client as _get_redis_client

# Re-export the existing get_redis_client function
get_redis_client = _get_redis_client

__all__ = ["get_redis_client"]
