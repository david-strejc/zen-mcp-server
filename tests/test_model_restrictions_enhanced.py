"""
Tests for enhanced model restriction features (blocking and patterns)
"""

import os
from unittest.mock import patch

import pytest

from providers.base import ProviderType
from utils.model_restrictions import ModelRestrictionService


class TestEnhancedModelRestrictions:
    """Test suite for enhanced model restriction features."""

    def test_blocked_models_basic(self):
        """Test that BLOCKED_MODELS environment variable blocks specific models."""
        with patch.dict(os.environ, {"BLOCKED_MODELS": "gpt-4,claude-opus,gemini-ultra"}):
            service = ModelRestrictionService()
            
            # Check blocked models are loaded
            assert "gpt-4" in service.blocked_models
            assert "claude-opus" in service.blocked_models
            assert "gemini-ultra" in service.blocked_models
            
            # These models should be blocked regardless of provider
            assert not service.is_allowed(ProviderType.OPENAI, "gpt-4")
            assert not service.is_allowed(ProviderType.OPENROUTER, "claude-opus")
            assert not service.is_allowed(ProviderType.GOOGLE, "gemini-ultra")
            
            # Other models should still be allowed
            assert service.is_allowed(ProviderType.OPENAI, "gpt-5")
            assert service.is_allowed(ProviderType.GOOGLE, "gemini-2.5-flash")

    def test_disabled_patterns(self):
        """Test that DISABLED_MODEL_PATTERNS blocks models matching patterns."""
        with patch.dict(os.environ, {"DISABLED_MODEL_PATTERNS": "claude,anthropic,gpt-3"}):
            service = ModelRestrictionService()
            
            # Check patterns are loaded
            assert "claude" in service.disabled_patterns
            assert "anthropic" in service.disabled_patterns
            assert "gpt-3" in service.disabled_patterns
            
            # Models matching patterns should be blocked
            assert not service.is_allowed(ProviderType.OPENROUTER, "claude-opus")
            assert not service.is_allowed(ProviderType.OPENROUTER, "claude-3-haiku")
            assert not service.is_allowed(ProviderType.OPENROUTER, "anthropic/claude-sonnet")
            assert not service.is_allowed(ProviderType.OPENAI, "gpt-3.5-turbo")
            
            # Models not matching patterns should be allowed
            assert service.is_allowed(ProviderType.OPENAI, "gpt-4")
            assert service.is_allowed(ProviderType.OPENAI, "gpt-5")
            assert service.is_allowed(ProviderType.OPENROUTER, "mistral-large")

    def test_pattern_priority_over_allowed(self):
        """Test that patterns and blocks override allowed lists."""
        env_vars = {
            "OPENROUTER_ALLOWED_MODELS": "claude-opus,mistral,llama",
            "DISABLED_MODEL_PATTERNS": "claude"
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            
            # claude-opus is in allowed list but matches disabled pattern
            assert not service.is_allowed(ProviderType.OPENROUTER, "claude-opus")
            
            # Other allowed models work fine
            assert service.is_allowed(ProviderType.OPENROUTER, "mistral")
            assert service.is_allowed(ProviderType.OPENROUTER, "llama")

    def test_blocked_models_override_allowed(self):
        """Test that explicitly blocked models override allowed lists."""
        env_vars = {
            "OPENAI_ALLOWED_MODELS": "gpt-4,gpt-5,o3-mini",
            "BLOCKED_MODELS": "gpt-4"
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            
            # gpt-4 is in allowed list but explicitly blocked
            assert not service.is_allowed(ProviderType.OPENAI, "gpt-4")
            
            # Other allowed models work fine
            assert service.is_allowed(ProviderType.OPENAI, "gpt-5")
            assert service.is_allowed(ProviderType.OPENAI, "o3-mini")

    def test_case_insensitive_matching(self):
        """Test that all matching is case-insensitive."""
        env_vars = {
            "BLOCKED_MODELS": "GPT-4,Claude-Opus",
            "DISABLED_MODEL_PATTERNS": "ANTHROPIC"
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            
            # Should match regardless of case
            assert not service.is_allowed(ProviderType.OPENAI, "gpt-4")
            assert not service.is_allowed(ProviderType.OPENAI, "GPT-4")
            assert not service.is_allowed(ProviderType.OPENROUTER, "claude-opus")
            assert not service.is_allowed(ProviderType.OPENROUTER, "CLAUDE-OPUS")
            assert not service.is_allowed(ProviderType.OPENROUTER, "anthropic/claude")
            assert not service.is_allowed(ProviderType.OPENROUTER, "ANTHROPIC/CLAUDE")

    def test_empty_or_whitespace_values(self):
        """Test that empty or whitespace values are handled correctly."""
        env_vars = {
            "BLOCKED_MODELS": "  ,  , gpt-4,  ,  ",
            "DISABLED_MODEL_PATTERNS": " claude ,  , , anthropic "
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            
            # Should only have the non-empty values
            assert service.blocked_models == {"gpt-4"}
            assert service.disabled_patterns == ["claude", "anthropic"]

    def test_no_restrictions_default(self):
        """Test that when no restrictions are set, all models are allowed."""
        with patch.dict(os.environ, {}, clear=True):
            service = ModelRestrictionService()
            
            # Everything should be allowed
            assert service.is_allowed(ProviderType.OPENAI, "gpt-4")
            assert service.is_allowed(ProviderType.OPENAI, "gpt-5")
            assert service.is_allowed(ProviderType.GOOGLE, "gemini-pro")
            assert service.is_allowed(ProviderType.OPENROUTER, "claude-opus")
            assert service.is_allowed(ProviderType.OPENROUTER, "anthropic/claude")

    def test_combined_restrictions(self):
        """Test combining allowed lists, blocked models, and patterns."""
        env_vars = {
            "OPENAI_ALLOWED_MODELS": "gpt-4,gpt-5,o3-mini",
            "GOOGLE_ALLOWED_MODELS": "gemini-2.5-flash,gemini-2.5-pro",
            "BLOCKED_MODELS": "gpt-4",
            "DISABLED_MODEL_PATTERNS": "claude,anthropic"
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            
            # OpenAI: gpt-4 blocked explicitly
            assert not service.is_allowed(ProviderType.OPENAI, "gpt-4")
            assert service.is_allowed(ProviderType.OPENAI, "gpt-5")
            assert service.is_allowed(ProviderType.OPENAI, "o3-mini")
            assert not service.is_allowed(ProviderType.OPENAI, "gpt-3.5")  # Not in allowed list
            
            # Google: only allowed models
            assert service.is_allowed(ProviderType.GOOGLE, "gemini-2.5-flash")
            assert service.is_allowed(ProviderType.GOOGLE, "gemini-2.5-pro")
            assert not service.is_allowed(ProviderType.GOOGLE, "gemini-ultra")  # Not in allowed list
            
            # OpenRouter: no allowed list, but patterns apply
            assert not service.is_allowed(ProviderType.OPENROUTER, "claude-opus")  # Pattern match
            assert not service.is_allowed(ProviderType.OPENROUTER, "anthropic/claude")  # Pattern match
            assert service.is_allowed(ProviderType.OPENROUTER, "mistral-large")  # No restrictions

    def test_restriction_summary(self):
        """Test that get_restriction_summary includes all restriction types."""
        env_vars = {
            "OPENAI_ALLOWED_MODELS": "gpt-5,o3-mini",
            "BLOCKED_MODELS": "gpt-4,claude-opus",
            "DISABLED_MODEL_PATTERNS": "anthropic,gpt-3"
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            summary = service.get_restriction_summary()
            
            # Check allowed models
            assert summary["openai"] == ["gpt-5", "o3-mini"]
            
            # Check blocked models
            assert "blocked_models" in summary
            assert summary["blocked_models"] == ["claude-opus", "gpt-4"]
            
            # Check disabled patterns
            assert "disabled_patterns" in summary
            assert summary["disabled_patterns"] == ["anthropic", "gpt-3"]

    def test_filter_models_with_blocks_and_patterns(self):
        """Test that filter_models respects blocks and patterns."""
        env_vars = {
            "BLOCKED_MODELS": "gpt-4",
            "DISABLED_MODEL_PATTERNS": "claude"
        }
        
        with patch.dict(os.environ, env_vars):
            service = ModelRestrictionService()
            
            models = ["gpt-4", "gpt-5", "claude-opus", "mistral-large"]
            filtered = service.filter_models(ProviderType.OPENROUTER, models)
            
            # Should only have gpt-5 and mistral-large
            assert filtered == ["gpt-5", "mistral-large"]