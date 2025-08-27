"""
Enhanced LLM integrations for SafePath Filter - Generation 1.

Advanced wrapper implementations for popular LLM frameworks with real-time
filtering, comprehensive monitoring, and adaptive safety controls.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from .core import SafePathFilter
from .models import FilterRequest, FilterResult, SafetyLevel, FilterConfig
from .exceptions import FilterError


logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for LLM integrations."""
    
    filter_input: bool = True
    filter_output: bool = True
    filter_streaming: bool = True
    safety_level: SafetyLevel = SafetyLevel.BALANCED
    max_retry_attempts: int = 3
    timeout_seconds: int = 30
    enable_metrics: bool = True
    log_all_requests: bool = False
    block_on_filter: bool = True


class BaseLLMIntegration(ABC):
    """Base class for LLM integrations."""
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        self.filter_config = FilterConfig(
            safety_level=self.config.safety_level,
            log_filtered=self.config.log_all_requests
        )
        self.safepath_filter = SafePathFilter(self.filter_config)
        self.metrics = {
            "total_requests": 0,
            "filtered_inputs": 0,
            "filtered_outputs": 0,
            "blocked_requests": 0,
            "avg_latency_ms": 0.0,
            "errors": 0
        }
        
    @abstractmethod
    def wrap_llm(self, llm: Any) -> Any:
        """Wrap an LLM instance with SafePath filtering."""
        pass
    
    def _filter_content(self, content: str, content_type: str = "input") -> FilterResult:
        """Filter content and update metrics."""
        start_time = time.time()
        
        try:
            request = FilterRequest(
                content=content,
                safety_level=self.config.safety_level,
                metadata={"content_type": content_type}
            )
            
            result = self.safepath_filter.filter(request)
            
            # Update metrics
            self.metrics["total_requests"] += 1
            if result.was_filtered:
                if content_type == "input":
                    self.metrics["filtered_inputs"] += 1
                elif content_type == "output":
                    self.metrics["filtered_outputs"] += 1
                
                if not result.safety_score.is_safe and self.config.block_on_filter:
                    self.metrics["blocked_requests"] += 1
            
            processing_time = (time.time() - start_time) * 1000
            self.metrics["avg_latency_ms"] = (
                (self.metrics["avg_latency_ms"] * (self.metrics["total_requests"] - 1) + processing_time) /
                self.metrics["total_requests"]
            )
            
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Content filtering failed: {e}")
            raise FilterError(f"Content filtering failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        return self.metrics.copy()


class OpenAIIntegration(BaseLLMIntegration):
    """Enhanced OpenAI integration with comprehensive safety filtering."""
    
    def wrap_llm(self, client: Any) -> Any:
        """Wrap OpenAI client with SafePath filtering."""
        
        class SafeOpenAIWrapper:
            def __init__(self, client, integration):
                self.client = client
                self.integration = integration
                self.original_create = client.chat.completions.create
                
                # Wrap the create method
                client.chat.completions.create = self._safe_create
            
            def _safe_create(self, **kwargs):
                """Safe wrapper for OpenAI chat completions."""
                messages = kwargs.get("messages", [])
                filtered_messages = []
                
                # Filter input messages
                for message in messages:
                    if "content" in message and message["content"]:
                        if self.integration.config.filter_input:
                            filter_result = self.integration._filter_content(
                                message["content"], "input"
                            )
                            
                            if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                                raise FilterError(
                                    f"Input blocked due to safety concerns: {filter_result.filter_reasons}",
                                    safety_score=filter_result.safety_score.overall_score
                                )
                            
                            message = message.copy()
                            message["content"] = filter_result.filtered_content
                        
                        filtered_messages.append(message)
                
                kwargs["messages"] = filtered_messages
                
                # Call original OpenAI API
                response = self.original_create(**kwargs)
                
                # Filter output if enabled
                if self.integration.config.filter_output and hasattr(response, 'choices'):
                    for choice in response.choices:
                        if hasattr(choice.message, 'content') and choice.message.content:
                            filter_result = self.integration._filter_content(
                                choice.message.content, "output"
                            )
                            
                            if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                                choice.message.content = "[CONTENT FILTERED: Safety violation detected]"
                            else:
                                choice.message.content = filter_result.filtered_content
                
                return response
            
            def __getattr__(self, name):
                """Delegate other attributes to the original client."""
                return getattr(self.client, name)
        
        return SafeOpenAIWrapper(client, self)


class LangChainIntegration(BaseLLMIntegration):
    """Enhanced LangChain integration with callback-based filtering."""
    
    def wrap_llm(self, llm: Any) -> Any:
        """Wrap LangChain LLM with SafePath filtering."""
        
        class SafeLangChainWrapper:
            def __init__(self, llm, integration):
                self.llm = llm
                self.integration = integration
                self.original_call = llm._call if hasattr(llm, '_call') else llm.__call__
                
                # Override call method
                if hasattr(llm, '_call'):
                    llm._call = self._safe_call
                else:
                    llm.__call__ = self._safe_call
            
            def _safe_call(self, prompt: str, **kwargs) -> str:
                """Safe wrapper for LangChain LLM calls."""
                # Filter input prompt
                if self.integration.config.filter_input:
                    filter_result = self.integration._filter_content(prompt, "input")
                    
                    if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                        raise FilterError(
                            f"Input blocked: {filter_result.filter_reasons}",
                            safety_score=filter_result.safety_score.overall_score
                        )
                    
                    filtered_prompt = filter_result.filtered_content
                else:
                    filtered_prompt = prompt
                
                # Call original LLM
                response = self.original_call(filtered_prompt, **kwargs)
                
                # Filter output
                if self.integration.config.filter_output and isinstance(response, str):
                    filter_result = self.integration._filter_content(response, "output")
                    
                    if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                        return "[CONTENT FILTERED: Safety violation detected]"
                    else:
                        return filter_result.filtered_content
                
                return response
            
            def __getattr__(self, name):
                """Delegate other attributes to the original LLM."""
                return getattr(self.llm, name)
        
        return SafeLangChainWrapper(llm, self)


class AutoGenIntegration(BaseLLMIntegration):
    """Enhanced AutoGen integration with multi-agent safety."""
    
    def wrap_llm(self, agent: Any) -> Any:
        """Wrap AutoGen agent with SafePath filtering."""
        
        class SafeAutoGenWrapper:
            def __init__(self, agent, integration):
                self.agent = agent
                self.integration = integration
                
                # Store original methods
                if hasattr(agent, 'generate_reply'):
                    self.original_generate_reply = agent.generate_reply
                    agent.generate_reply = self._safe_generate_reply
                
                if hasattr(agent, 'send'):
                    self.original_send = agent.send
                    agent.send = self._safe_send
            
            def _safe_generate_reply(self, messages, **kwargs):
                """Safe wrapper for AutoGen reply generation."""
                # Filter incoming messages
                if self.integration.config.filter_input:
                    filtered_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and "content" in msg:
                            filter_result = self.integration._filter_content(
                                msg["content"], "input"
                            )
                            
                            if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                                # Skip unsafe messages or replace with safety notice
                                safe_msg = msg.copy()
                                safe_msg["content"] = "[MESSAGE FILTERED: Safety violation]"
                                filtered_messages.append(safe_msg)
                            else:
                                safe_msg = msg.copy()
                                safe_msg["content"] = filter_result.filtered_content
                                filtered_messages.append(safe_msg)
                        else:
                            filtered_messages.append(msg)
                    messages = filtered_messages
                
                # Generate reply
                reply = self.original_generate_reply(messages, **kwargs)
                
                # Filter output reply
                if self.integration.config.filter_output and reply:
                    if isinstance(reply, str):
                        filter_result = self.integration._filter_content(reply, "output")
                        
                        if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                            return "[REPLY FILTERED: Safety violation detected]"
                        else:
                            return filter_result.filtered_content
                    
                    elif isinstance(reply, dict) and "content" in reply:
                        filter_result = self.integration._filter_content(
                            reply["content"], "output"
                        )
                        
                        if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                            reply = reply.copy()
                            reply["content"] = "[REPLY FILTERED: Safety violation detected]"
                        else:
                            reply = reply.copy()
                            reply["content"] = filter_result.filtered_content
                
                return reply
            
            def _safe_send(self, message, recipient, **kwargs):
                """Safe wrapper for AutoGen message sending."""
                # Filter outgoing message
                if self.integration.config.filter_output and isinstance(message, str):
                    filter_result = self.integration._filter_content(message, "output")
                    
                    if not filter_result.safety_score.is_safe and self.integration.config.block_on_filter:
                        message = "[MESSAGE FILTERED: Safety violation detected]"
                    else:
                        message = filter_result.filtered_content
                
                return self.original_send(message, recipient, **kwargs)
            
            def __getattr__(self, name):
                """Delegate other attributes to the original agent."""
                return getattr(self.agent, name)
        
        return SafeAutoGenWrapper(agent, self)


class StreamingIntegration:
    """Enhanced streaming response filtering."""
    
    def __init__(self, safepath_filter: SafePathFilter, buffer_size: int = 1024):
        self.safepath_filter = safepath_filter
        self.buffer_size = buffer_size
        self.current_buffer = ""
        self.accumulated_content = ""
        
    async def filter_stream(self, stream_generator):
        """Filter streaming content in real-time."""
        try:
            async for chunk in stream_generator:
                # Accumulate content
                if isinstance(chunk, str):
                    content = chunk
                elif hasattr(chunk, 'choices') and chunk.choices:
                    # OpenAI-style streaming
                    content = chunk.choices[0].delta.content or ""
                else:
                    content = str(chunk)
                
                self.current_buffer += content
                self.accumulated_content += content
                
                # Process buffer when it reaches threshold
                if len(self.current_buffer) >= self.buffer_size:
                    filtered_chunk = await self._process_buffer()
                    if filtered_chunk:
                        yield filtered_chunk
                
                # Also yield individual chunks for responsiveness
                if content:
                    # Quick filter check for obvious problems
                    if any(term in content.lower() for term in ['bomb', 'kill', 'poison', 'hack']):
                        # Skip potentially harmful content
                        continue
                    yield content
            
            # Process remaining buffer
            if self.current_buffer:
                filtered_chunk = await self._process_buffer()
                if filtered_chunk:
                    yield filtered_chunk
            
            # Final safety check on complete content
            await self._final_safety_check()
            
        except Exception as e:
            logger.error(f"Stream filtering failed: {e}")
            yield "[STREAM FILTERED: Safety violation detected]"
    
    async def _process_buffer(self) -> str:
        """Process accumulated buffer content."""
        if not self.current_buffer.strip():
            return ""
        
        try:
            request = FilterRequest(
                content=self.current_buffer,
                metadata={"stream_chunk": True}
            )
            
            result = self.safepath_filter.filter(request)
            
            if result.was_filtered and not result.safety_score.is_safe:
                logger.warning(f"Streaming content filtered: {result.filter_reasons}")
                self.current_buffer = ""
                return "[CONTENT FILTERED]"
            
            filtered_content = result.filtered_content
            self.current_buffer = ""
            return filtered_content
            
        except Exception as e:
            logger.error(f"Buffer processing failed: {e}")
            self.current_buffer = ""
            return "[PROCESSING ERROR]"
    
    async def _final_safety_check(self) -> None:
        """Perform final safety check on complete accumulated content."""
        if not self.accumulated_content.strip():
            return
        
        try:
            request = FilterRequest(
                content=self.accumulated_content,
                metadata={"final_check": True}
            )
            
            result = self.safepath_filter.filter(request)
            
            if result.was_filtered and not result.safety_score.is_safe:
                logger.warning(
                    f"Final safety check failed for stream: {result.filter_reasons}"
                )
                # Could implement additional remediation here
                
        except Exception as e:
            logger.error(f"Final safety check failed: {e}")


class IntegrationFactory:
    """Factory for creating LLM integrations."""
    
    INTEGRATIONS = {
        "openai": OpenAIIntegration,
        "langchain": LangChainIntegration,
        "autogen": AutoGenIntegration,
    }
    
    @classmethod
    def create_integration(
        self, 
        integration_type: str, 
        config: IntegrationConfig = None
    ) -> BaseLLMIntegration:
        """Create an integration instance."""
        if integration_type not in self.INTEGRATIONS:
            available = ", ".join(self.INTEGRATIONS.keys())
            raise ValueError(f"Unknown integration type '{integration_type}'. Available: {available}")
        
        integration_class = self.INTEGRATIONS[integration_type]
        return integration_class(config)
    
    @classmethod
    def wrap_llm(
        self,
        llm: Any,
        integration_type: str = "auto",
        config: IntegrationConfig = None
    ) -> Any:
        """Auto-detect and wrap LLM with appropriate integration."""
        if integration_type == "auto":
            integration_type = self._detect_llm_type(llm)
        
        integration = self.create_integration(integration_type, config)
        return integration.wrap_llm(llm)
    
    @classmethod
    def _detect_llm_type(self, llm: Any) -> str:
        """Auto-detect LLM type based on class name and attributes."""
        class_name = llm.__class__.__name__.lower()
        module_name = llm.__class__.__module__.lower()
        
        if "openai" in class_name or "openai" in module_name:
            return "openai"
        elif "langchain" in class_name or "langchain" in module_name:
            return "langchain"
        elif "autogen" in class_name or "autogen" in module_name:
            return "autogen"
        else:
            # Default to LangChain-style integration
            return "langchain"


# Convenience functions for quick integration

def wrap_openai_client(client, safety_level: SafetyLevel = SafetyLevel.BALANCED):
    """Quickly wrap OpenAI client with SafePath filtering."""
    config = IntegrationConfig(safety_level=safety_level)
    integration = OpenAIIntegration(config)
    return integration.wrap_llm(client)


def wrap_langchain_llm(llm, safety_level: SafetyLevel = SafetyLevel.BALANCED):
    """Quickly wrap LangChain LLM with SafePath filtering."""
    config = IntegrationConfig(safety_level=safety_level)
    integration = LangChainIntegration(config)
    return integration.wrap_llm(llm)


def wrap_autogen_agent(agent, safety_level: SafetyLevel = SafetyLevel.BALANCED):
    """Quickly wrap AutoGen agent with SafePath filtering."""
    config = IntegrationConfig(safety_level=safety_level)
    integration = AutoGenIntegration(config)
    return integration.wrap_llm(agent)


def create_safe_streaming_filter(safepath_filter: SafePathFilter = None):
    """Create a streaming content filter."""
    if safepath_filter is None:
        safepath_filter = SafePathFilter()
    return StreamingIntegration(safepath_filter)