#!/usr/bin/env python3
"""
Automated backup system for CoT SafePath Filter.

This script provides automated backup capabilities for:
- Database snapshots
- Configuration files
- Log archives
- Model weights and checkpoints
- Application state
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class BackupManager:
    """Manages automated backups for the SafePath application."""
    
    def __init__(self, config: Dict):
        """Initialize backup manager with configuration."""
        self.config = config
        self.s3_client = None
        self.backup_root = Path(config.get('backup_root', '/tmp/safepath_backups'))
        self.retention_days = config.get('retention_days', 30)
        
        # Initialize S3 client if configured
        if config.get('s3_enabled', False):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key'),
                region_name=config.get('aws_region', 'us-east-1')
            )
    
    async def create_full_backup(self) -> Dict[str, str]:
        """Create a complete system backup."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.backup_root / f"full_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting full backup", backup_dir=str(backup_dir))
        
        results = {}
        
        try:
            # Database backup
            if self.config.get('backup_database', True):
                db_backup = await self._backup_database(backup_dir)
                results['database'] = db_backup
            
            # Configuration backup
            if self.config.get('backup_config', True):
                config_backup = await self._backup_configuration(backup_dir)
                results['configuration'] = config_backup
            
            # Logs backup
            if self.config.get('backup_logs', True):
                logs_backup = await self._backup_logs(backup_dir)
                results['logs'] = logs_backup
            
            # Model weights backup
            if self.config.get('backup_models', True):
                models_backup = await self._backup_models(backup_dir)
                results['models'] = models_backup
            
            # Application state backup
            if self.config.get('backup_state', True):
                state_backup = await self._backup_application_state(backup_dir)
                results['state'] = state_backup
            
            # Create backup manifest
            manifest_path = await self._create_backup_manifest(backup_dir, results)
            results['manifest'] = manifest_path
            
            # Compress backup
            if self.config.get('compress_backups', True):
                compressed_path = await self._compress_backup(backup_dir)
                results['compressed'] = compressed_path
            
            # Upload to S3 if configured
            if self.s3_client and self.config.get('upload_to_s3', False):
                s3_path = await self._upload_to_s3(compressed_path or backup_dir)
                results['s3_path'] = s3_path
            
            logger.info("Full backup completed successfully", results=results)
            
        except Exception as e:
            logger.error("Backup failed", error=str(e), exc_info=True)
            raise
        
        finally:
            # Cleanup old backups
            await self._cleanup_old_backups()
        
        return results
    
    async def _backup_database(self, backup_dir: Path) -> str:
        """Backup database using appropriate method."""
        db_backup_dir = backup_dir / "database"
        db_backup_dir.mkdir(exist_ok=True)
        
        db_type = self.config.get('database_type', 'sqlite')
        
        if db_type == 'sqlite':
            # SQLite backup
            db_path = self.config.get('database_path', 'safepath.db')
            if Path(db_path).exists():
                backup_path = db_backup_dir / f"safepath_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2(db_path, backup_path)
                logger.info("SQLite database backed up", path=str(backup_path))
                return str(backup_path)
        
        elif db_type == 'postgresql':
            # PostgreSQL backup using pg_dump
            backup_path = db_backup_dir / f"postgresql_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            cmd = [
                'pg_dump',
                '--host', self.config.get('db_host', 'localhost'),
                '--port', str(self.config.get('db_port', 5432)),
                '--username', self.config.get('db_user', 'safepath'),
                '--dbname', self.config.get('db_name', 'safepath'),
                '--file', str(backup_path),
                '--verbose',
                '--no-password'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("PostgreSQL database backed up", path=str(backup_path))
                return str(backup_path)
            else:
                raise Exception(f"PostgreSQL backup failed: {result.stderr}")
        
        return ""
    
    async def _backup_configuration(self, backup_dir: Path) -> str:
        """Backup configuration files."""
        config_backup_dir = backup_dir / "configuration"
        config_backup_dir.mkdir(exist_ok=True)
        
        config_files = [
            'pyproject.toml',
            'docker-compose.yml',
            'docker-compose.dev.yml',
            '.env.example',
            'Dockerfile',
            'monitoring/prometheus/prometheus.yml',
            'monitoring/grafana/provisioning/'
        ]
        
        for config_file in config_files:
            src_path = Path(config_file)
            if src_path.exists():
                if src_path.is_file():
                    dst_path = config_backup_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                elif src_path.is_dir():
                    dst_path = config_backup_dir / src_path.name
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        
        logger.info("Configuration files backed up", dir=str(config_backup_dir))
        return str(config_backup_dir)
    
    async def _backup_logs(self, backup_dir: Path) -> str:
        """Backup application logs."""
        logs_backup_dir = backup_dir / "logs"
        logs_backup_dir.mkdir(exist_ok=True)
        
        log_dirs = ['logs/', 'audit_logs/', '/var/log/safepath/']
        
        for log_dir in log_dirs:
            log_path = Path(log_dir)
            if log_path.exists() and log_path.is_dir():
                dst_path = logs_backup_dir / log_path.name
                shutil.copytree(log_path, dst_path, dirs_exist_ok=True)
        
        logger.info("Logs backed up", dir=str(logs_backup_dir))
        return str(logs_backup_dir)
    
    async def _backup_models(self, backup_dir: Path) -> str:
        """Backup ML models and weights."""
        models_backup_dir = backup_dir / "models"
        models_backup_dir.mkdir(exist_ok=True)
        
        model_dirs = ['models/', 'checkpoints/', '.cache/']
        
        for model_dir in model_dirs:
            model_path = Path(model_dir)
            if model_path.exists() and model_path.is_dir():
                dst_path = models_backup_dir / model_path.name
                shutil.copytree(model_path, dst_path, dirs_exist_ok=True)
        
        logger.info("Models backed up", dir=str(models_backup_dir))
        return str(models_backup_dir)
    
    async def _backup_application_state(self, backup_dir: Path) -> str:
        """Backup application state and runtime data."""
        state_backup_dir = backup_dir / "state"
        state_backup_dir.mkdir(exist_ok=True)
        
        # Backup Redis state if configured
        if self.config.get('redis_enabled', False):
            await self._backup_redis_state(state_backup_dir)
        
        # Backup any persistent caches
        cache_dirs = ['.cache/', 'tmp/']
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if cache_path.exists() and cache_path.is_dir():
                dst_path = state_backup_dir / cache_path.name
                shutil.copytree(cache_path, dst_path, dirs_exist_ok=True)
        
        logger.info("Application state backed up", dir=str(state_backup_dir))
        return str(state_backup_dir)
    
    async def _backup_redis_state(self, state_dir: Path):
        """Backup Redis database."""
        redis_backup_path = state_dir / f"redis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rdb"
        
        cmd = [
            'redis-cli',
            '--rdb', str(redis_backup_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Redis state backed up", path=str(redis_backup_path))
        else:
            logger.warning("Redis backup failed", error=result.stderr)
    
    async def _create_backup_manifest(self, backup_dir: Path, results: Dict) -> str:
        """Create backup manifest with metadata."""
        manifest_path = backup_dir / "backup_manifest.json"
        
        manifest = {
            'backup_timestamp': datetime.now().isoformat(),
            'backup_type': 'full',
            'safepath_version': self.config.get('version', 'unknown'),
            'components_backed_up': list(results.keys()),
            'backup_size_bytes': self._calculate_directory_size(backup_dir),
            'retention_until': (datetime.now() + timedelta(days=self.retention_days)).isoformat(),
            'backup_integrity': await self._calculate_backup_checksums(backup_dir)
        }
        
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("Backup manifest created", path=str(manifest_path))
        return str(manifest_path)
    
    async def _compress_backup(self, backup_dir: Path) -> str:
        """Compress backup directory."""
        compressed_path = f"{backup_dir}.tar.gz"
        
        cmd = ['tar', '-czf', compressed_path, '-C', str(backup_dir.parent), backup_dir.name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Remove original directory after successful compression
            shutil.rmtree(backup_dir)
            logger.info("Backup compressed successfully", path=compressed_path)
            return compressed_path
        else:
            raise Exception(f"Compression failed: {result.stderr}")
    
    async def _upload_to_s3(self, backup_path: str) -> str:
        """Upload backup to S3."""
        bucket = self.config.get('s3_bucket')
        if not bucket:
            raise ValueError("S3 bucket not configured")
        
        backup_name = Path(backup_path).name
        s3_key = f"safepath-backups/{datetime.now().strftime('%Y/%m/%d')}/{backup_name}"
        
        try:
            self.s3_client.upload_file(backup_path, bucket, s3_key)
            logger.info("Backup uploaded to S3", bucket=bucket, key=s3_key)
            return f"s3://{bucket}/{s3_key}"
        except Exception as e:
            logger.error("S3 upload failed", error=str(e))
            raise
    
    async def _cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for backup_path in self.backup_root.glob("*"):
            if backup_path.is_file() or backup_path.is_dir():
                # Extract timestamp from backup name
                try:
                    timestamp_str = backup_path.name.split('_')[-2] + '_' + backup_path.name.split('_')[-1]
                    if backup_path.suffix:  # Remove extension
                        timestamp_str = timestamp_str.replace(backup_path.suffix, '')
                    
                    backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    if backup_date < cutoff_date:
                        if backup_path.is_dir():
                            shutil.rmtree(backup_path)
                        else:
                            backup_path.unlink()
                        logger.info("Old backup removed", path=str(backup_path))
                        
                except (ValueError, IndexError):
                    # Skip files that don't match expected naming pattern
                    continue
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                total_size += filepath.stat().st_size
        return total_size
    
    async def _calculate_backup_checksums(self, backup_dir: Path) -> Dict[str, str]:
        """Calculate checksums for backup integrity verification."""
        checksums = {}
        
        for file_path in backup_dir.rglob('*'):
            if file_path.is_file():
                import hashlib
                
                hash_md5 = hashlib.md5()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                
                relative_path = file_path.relative_to(backup_dir)
                checksums[str(relative_path)] = hash_md5.hexdigest()
        
        return checksums


async def main():
    """Main backup execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SafePath Backup Manager')
    parser.add_argument('--config', default='backup_config.json', help='Backup configuration file')
    parser.add_argument('--type', choices=['full', 'incremental'], default='full', help='Backup type')
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if Path(args.config).exists():
        import json
        with open(args.config) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'backup_root': '/tmp/safepath_backups',
            'retention_days': 30,
            'compress_backups': True,
            'backup_database': True,
            'backup_config': True,
            'backup_logs': True,
            'backup_models': True,
            'backup_state': True,
            's3_enabled': False
        }
    
    # Initialize backup manager
    backup_manager = BackupManager(config)
    
    try:
        if args.type == 'full':
            results = await backup_manager.create_full_backup()
            print(f"Backup completed successfully: {results}")
        else:
            print("Incremental backups not yet implemented")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Backup failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())