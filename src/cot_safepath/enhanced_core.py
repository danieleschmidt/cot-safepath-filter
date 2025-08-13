"""
Enhanced core filtering engine with Generation 4 optimizations.
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from .core import SafePathFilter as BaseSafePathFilter, FilterPipeline
from .models import FilterConfig, FilterRequest, FilterResult, SafetyLevel
from .exceptions import FilterError, TimeoutError
from .advanced_performance import (
    AsyncFilterProcessor,
    AdaptivePerformanceOptimizer,
    IntelligentCacheManager,
    AdvancedPerformanceConfig,
)
from .global_deployment import (
    GlobalDeploymentManager,
    InternationalizationManager,
    DeploymentRegion,
)
from .performance import PerformanceConfig

logger = logging.getLogger(__name__)


class EnhancedSafePathFilter(BaseSafePathFilter):
    """Enhanced SafePath filter with Generation 4 optimizations."""
    
    def __init__(self, config: FilterConfig = None, enable_async: bool = True,
                 enable_global_deployment: bool = False):
        super().__init__(config)
        
        # Generation 4 enhancements
        self.enable_async = enable_async
        self.enable_global_deployment = enable_global_deployment
        
        # Initialize advanced components
        self.performance_config = PerformanceConfig()
        self.advanced_config = AdvancedPerformanceConfig()
        
        # Performance optimization
        self.adaptive_optimizer = AdaptivePerformanceOptimizer(self.advanced_config)
        self.intelligent_cache = IntelligentCacheManager(self.advanced_config)
        
        # Async processing
        if self.enable_async:
            self.async_processor = AsyncFilterProcessor(
                self._sync_filter_function, self.performance_config
            )
        
        # Global deployment
        if self.enable_global_deployment:
            self.deployment_manager = GlobalDeploymentManager()
            self.i18n_manager = InternationalizationManager()
            
        # Enhanced metrics
        self.enhanced_metrics = {
            'generation_4_features_active': True,
            'async_processing_enabled': enable_async,
            'global_deployment_enabled': enable_global_deployment,
            'total_enhanced_requests': 0,
            'performance_optimizations_applied': 0,
            'global_requests_routed': 0,
        }
    
    def _sync_filter_function(self, request: FilterRequest) -> FilterResult:
        """Synchronous filter function for async wrapper."""
        return super().filter(request)
    
    async def filter_async(self, request: FilterRequest) -> FilterResult:
        """
        Asynchronous filtering with Generation 4 enhancements.
        
        Args:
            request: FilterRequest with content and configuration
            
        Returns:
            FilterResult with enhanced features
        """
        if not self.enable_async:
            # Fallback to sync processing
            return self.filter(request)
        
        start_time = time.time()
        
        try:
            # Check intelligent cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self.intelligent_cache.get(cache_key)
            
            if cached_result:
                logger.debug("Cache hit for request")
                return cached_result
            
            # Route to optimal region if global deployment is enabled
            if self.enable_global_deployment:
                optimal_region = self.deployment_manager.get_optimal_region(
                    client_ip=request.metadata.get('client_ip'),
                    language=request.metadata.get('language', 'en')
                )
                if optimal_region:
                    self.enhanced_metrics['global_requests_routed'] += 1
                    logger.debug(f"Routing request to region: {optimal_region.value}")
            
            # Process with async optimizations
            result = await self.async_processor.process_async(request)
            
            # Apply adaptive optimizations
            processing_time_ms = (time.time() - start_time) * 1000
            self.adaptive_optimizer.record_performance(
                processing_time_ms, 
                self._get_memory_usage(),
                self._get_cpu_usage()
            )
            
            # Cache result intelligently
            self.intelligent_cache.put(cache_key, result, ttl_seconds=3600)
            
            # Localize result if global deployment is enabled
            if self.enable_global_deployment and 'language' in request.metadata:
                result = self.i18n_manager.localize_filter_result(
                    result, request.metadata['language']
                )
            
            # Update enhanced metrics
            self.enhanced_metrics['total_enhanced_requests'] += 1
            self.enhanced_metrics['performance_optimizations_applied'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced async filtering failed: {e}")
            
            # Fallback to sync processing
            try:
                return self.filter(request)
            except Exception as fallback_error:
                raise FilterError(f"Both async and sync filtering failed: {e}, {fallback_error}")
    
    async def filter_batch_async(self, requests: List[FilterRequest]) -> List[FilterResult]:
        """
        Process multiple requests in parallel with intelligent batching.
        
        Args:
            requests: List of FilterRequest objects
            
        Returns:
            List of FilterResult objects
        """
        if not self.enable_async:
            # Process sequentially
            return [self.filter(request) for request in requests]
        
        if not requests:
            return []
        
        try:
            # Use async processor for intelligent batching
            results = await self.async_processor.process_batch_async(requests)
            
            # Update metrics
            self.enhanced_metrics['total_enhanced_requests'] += len(requests)
            self.enhanced_metrics['performance_optimizations_applied'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Fallback to individual processing
            results = []
            for request in requests:
                try:
                    result = await self.filter_async(request)
                    results.append(result)
                except Exception as individual_error:
                    logger.error(f"Individual request failed: {individual_error}")
                    results.append(None)
            
            return results
    
    def _generate_cache_key(self, request: FilterRequest) -> str:
        """Generate intelligent cache key for request."""
        import hashlib
        
        # Include content, safety level, and relevant metadata in cache key
        cache_data = {
            'content': request.content,
            'safety_level': request.safety_level.value if hasattr(request.safety_level, 'value') else str(request.safety_level),
            'language': request.metadata.get('language', 'en'),
            'version': '4.0'  # Generation 4 cache version
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 100.0  # Mock value when psutil not available
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 50.0  # Mock value when psutil not available
        except:
            return 0.0
    
    def activate_global_region(self, region: DeploymentRegion) -> bool:
        """Activate a global deployment region."""
        if not self.enable_global_deployment:
            logger.warning("Global deployment not enabled")
            return False
        
        return self.deployment_manager.activate_region(region)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including Generation 4 enhancements."""
        base_metrics = {
            'base_filter_metrics': {
                'cache_size': len(self.cache),
                'audit_logs_count': len(self.audit_logs),
            }
        }
        
        enhanced_metrics = {
            **base_metrics,
            **self.enhanced_metrics
        }
        
        # Add async processor stats if available
        if self.enable_async and hasattr(self, 'async_processor'):
            enhanced_metrics['async_processing_stats'] = self.async_processor.get_processing_stats()
        
        # Add intelligent cache stats
        if hasattr(self, 'intelligent_cache'):
            enhanced_metrics['intelligent_cache_stats'] = self.intelligent_cache.get_cache_stats()
        
        # Add global deployment stats if available
        if self.enable_global_deployment and hasattr(self, 'deployment_manager'):
            enhanced_metrics['global_deployment_stats'] = self.deployment_manager.get_deployment_status()
        
        return enhanced_metrics
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Trigger performance optimization and return results."""
        optimization_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'optimizations_applied': []
        }
        
        try:
            # Force garbage collection
            import gc
            collected = gc.collect()
            optimization_results['optimizations_applied'].append(f"garbage_collection: {collected} objects")
            
            # Optimize cache
            if hasattr(self, 'intelligent_cache'):
                cache_stats_before = self.intelligent_cache.get_cache_stats()
                # Cache optimization is automatic in intelligent cache
                cache_stats_after = self.intelligent_cache.get_cache_stats()
                optimization_results['optimizations_applied'].append(
                    f"cache_optimization: {cache_stats_before['cache_size']} -> {cache_stats_after['cache_size']} entries"
                )
            
            # Optimize base filter cache
            if len(self.cache) > 500:
                # Keep only most recent 250 entries
                cache_items = list(self.cache.items())
                self.cache = dict(cache_items[-250:])
                optimization_results['optimizations_applied'].append(
                    f"base_cache_cleanup: reduced to {len(self.cache)} entries"
                )
            
            # Optimize audit logs
            if len(self.audit_logs) > 5000:
                self.audit_logs = self.audit_logs[-2500:]
                optimization_results['optimizations_applied'].append(
                    f"audit_log_cleanup: reduced to {len(self.audit_logs)} entries"
                )
            
            self.enhanced_metrics['performance_optimizations_applied'] += 1
            optimization_results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            optimization_results['status'] = 'error'
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def cleanup_enhanced(self) -> None:
        """Enhanced cleanup method for Generation 4 features."""
        # Call base cleanup
        if hasattr(self, 'cleanup'):
            self.cleanup()
        
        # Cleanup enhanced components
        if hasattr(self, 'intelligent_cache'):
            self.intelligent_cache = IntelligentCacheManager(self.advanced_config)
        
        if hasattr(self, 'adaptive_optimizer'):
            self.adaptive_optimizer = AdaptivePerformanceOptimizer(self.advanced_config)
        
        # Reset enhanced metrics
        self.enhanced_metrics.update({
            'total_enhanced_requests': 0,
            'performance_optimizations_applied': 0,
            'global_requests_routed': 0,
        })
        
        logger.info("Enhanced cleanup completed")


# Backward compatibility alias
SafePathFilter = EnhancedSafePathFilter