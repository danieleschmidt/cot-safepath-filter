#!/usr/bin/env python3
"""
Security scanner for CoT SafePath Filter
Comprehensive security analysis and vulnerability detection
"""

import os
import json
import subprocess
import tempfile
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """Security finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(Enum):
    """Security finding categories."""
    VULNERABILITY = "vulnerability"
    SECRET = "secret"
    CODE_QUALITY = "code_quality"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    COMPLIANCE = "compliance"


@dataclass
class SecurityFinding:
    """Individual security finding."""
    id: str
    title: str
    description: str
    severity: Severity
    category: Category
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'category': self.category.value,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'code_snippet': self.code_snippet,
            'recommendation': self.recommendation,
            'references': self.references,
            'cwe_id': self.cwe_id,
            'cvss_score': self.cvss_score,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SecurityReport:
    """Security scan report containing all findings."""
    scan_id: str
    timestamp: datetime
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scan_id': self.scan_id,
            'timestamp': self.timestamp.isoformat(),
            'findings': [f.to_dict() for f in self.findings],
            'summary': self.summary,
            'metadata': self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class SecurityScanner:
    """Comprehensive security scanner for the SafePath project."""

    def __init__(self, project_root: str, config: Optional[Dict[str, Any]] = None):
        self.project_root = Path(project_root)
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scanner configuration
        self.exclude_patterns = self.config.get('exclude_patterns', [
            '__pycache__',
            '.git',
            '.venv',
            'venv',
            'node_modules',
            '.pytest_cache',
            '.mypy_cache',
            '.ruff_cache'
        ])

    def scan_all(self) -> SecurityReport:
        """Run all security scans and return comprehensive report."""
        scan_id = f"scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        findings = []
        
        # Run all individual scans
        findings.extend(self.scan_bandit())
        findings.extend(self.scan_safety())
        findings.extend(self.scan_semgrep())
        findings.extend(self.scan_secrets())
        findings.extend(self.scan_dependencies())
        findings.extend(self.scan_docker_security())
        findings.extend(self.scan_configuration())
        findings.extend(self.scan_code_quality())
        
        # Generate summary
        summary = self._generate_summary(findings)
        
        return SecurityReport(
            scan_id=scan_id,
            timestamp=datetime.now(timezone.utc),
            findings=findings,
            summary=summary,
            metadata={
                'project_root': str(self.project_root),
                'total_files_scanned': self._count_source_files(),
                'scanners_used': ['bandit', 'safety', 'semgrep', 'secrets', 'dependencies', 'docker', 'configuration', 'code_quality']
            }
        )

    def scan_bandit(self) -> List[SecurityFinding]:
        """Run Bandit security linter for Python code."""
        findings = []
        
        try:
            # Run bandit
            cmd = [
                'bandit',
                '-r', str(self.project_root / 'src'),
                '-f', 'json',
                '--skip', 'B101',  # Skip assert_used test
                '--exclude', ','.join(self.exclude_patterns)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_output = json.loads(result.stdout)
                    
                    for issue in bandit_output.get('results', []):
                        severity_map = {
                            'LOW': Severity.LOW,
                            'MEDIUM': Severity.MEDIUM,
                            'HIGH': Severity.HIGH
                        }
                        
                        finding = SecurityFinding(
                            id=f"bandit_{issue.get('test_id', 'unknown')}_{hash(issue.get('filename', '') + str(issue.get('line_number', 0)))}",
                            title=f"Bandit: {issue.get('test_name', 'Security Issue')}",
                            description=issue.get('issue_text', 'No description'),
                            severity=severity_map.get(issue.get('issue_severity', 'MEDIUM'), Severity.MEDIUM),
                            category=Category.VULNERABILITY,
                            file_path=issue.get('filename'),
                            line_number=issue.get('line_number'),
                            code_snippet=issue.get('code'),
                            recommendation=f"Review and fix the security issue identified by Bandit test {issue.get('test_id')}",
                            references=[f"https://bandit.readthedocs.io/en/latest/plugins/{issue.get('test_id', '').lower()}.html"],
                            cwe_id=issue.get('more_info', '').split('/')[-1] if 'cwe' in issue.get('more_info', '') else None
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse Bandit output: {e}")
            
        except FileNotFoundError:
            self.logger.warning("Bandit not found, skipping security scan")
        except Exception as e:
            self.logger.error(f"Bandit scan failed: {e}")
        
        return findings

    def scan_safety(self) -> List[SecurityFinding]:
        """Run Safety scanner for known vulnerabilities in dependencies."""
        findings = []
        
        try:
            # Run safety check
            cmd = ['safety', 'check', '--json']
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode in [0, 64]:  # 0 = no issues, 64 = vulnerabilities found
                try:
                    if result.stdout.strip():
                        safety_output = json.loads(result.stdout)
                        
                        for vuln in safety_output:
                            finding = SecurityFinding(
                                id=f"safety_{vuln.get('id', 'unknown')}",
                                title=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                                description=vuln.get('advisory', 'Known security vulnerability in dependency'),
                                severity=Severity.HIGH,  # All Safety issues are considered high severity
                                category=Category.DEPENDENCY,
                                recommendation=f"Update {vuln.get('package')} to version {vuln.get('vulnerable_spec', 'latest')} or higher",
                                references=[f"https://pyup.io/vulnerabilities/{vuln.get('id', '')}/"],
                                cvss_score=vuln.get('cvss')
                            )
                            findings.append(finding)
                            
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse Safety output: {e}")
            
        except FileNotFoundError:
            self.logger.warning("Safety not found, skipping dependency vulnerability scan")
        except Exception as e:
            self.logger.error(f"Safety scan failed: {e}")
        
        return findings

    def scan_semgrep(self) -> List[SecurityFinding]:
        """Run Semgrep for advanced security pattern detection."""
        findings = []
        
        try:
            # Run semgrep
            cmd = [
                'semgrep',
                '--config=auto',
                '--json',
                '--exclude', ','.join(self.exclude_patterns),
                str(self.project_root)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                try:
                    semgrep_output = json.loads(result.stdout)
                    
                    for issue in semgrep_output.get('results', []):
                        severity_map = {
                            'ERROR': Severity.HIGH,
                            'WARNING': Severity.MEDIUM,
                            'INFO': Severity.LOW
                        }
                        
                        finding = SecurityFinding(
                            id=f"semgrep_{issue.get('check_id', 'unknown').replace('.', '_')}_{hash(issue.get('path', '') + str(issue.get('start', {}).get('line', 0)))}",
                            title=f"Semgrep: {issue.get('check_id', 'Security Pattern')}",
                            description=issue.get('message', 'Security pattern detected'),
                            severity=severity_map.get(issue.get('extra', {}).get('severity', 'WARNING'), Severity.MEDIUM),
                            category=Category.VULNERABILITY,
                            file_path=issue.get('path'),
                            line_number=issue.get('start', {}).get('line'),
                            code_snippet=issue.get('extra', {}).get('lines'),
                            recommendation="Review the detected security pattern and apply appropriate fixes",
                            references=issue.get('extra', {}).get('references', [])
                        )
                        findings.append(finding)
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse Semgrep output: {e}")
            
        except FileNotFoundError:
            self.logger.warning("Semgrep not found, skipping advanced security scan")
        except Exception as e:
            self.logger.error(f"Semgrep scan failed: {e}")
        
        return findings

    def scan_secrets(self) -> List[SecurityFinding]:
        """Scan for exposed secrets and credentials."""
        findings = []
        
        try:
            # Run detect-secrets
            cmd = [
                'detect-secrets',
                'scan',
                '--all-files',
                '--force-use-all-plugins',
                str(self.project_root)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                try:
                    secrets_output = json.loads(result.stdout)
                    
                    for file_path, secrets in secrets_output.get('results', {}).items():
                        for secret in secrets:
                            finding = SecurityFinding(
                                id=f"secret_{secret.get('type', 'unknown')}_{hash(file_path + str(secret.get('line_number', 0)))}",
                                title=f"Exposed Secret: {secret.get('type', 'Unknown Type')}",
                                description=f"Potential secret detected: {secret.get('type', 'unknown type')}",
                                severity=Severity.CRITICAL,
                                category=Category.SECRET,
                                file_path=file_path,
                                line_number=secret.get('line_number'),
                                recommendation="Remove or properly secure the exposed secret",
                                references=["https://github.com/Yelp/detect-secrets"]
                            )
                            findings.append(finding)
                            
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse detect-secrets output: {e}")
            
        except FileNotFoundError:
            # Fallback to basic secret patterns
            findings.extend(self._scan_secrets_basic())
        except Exception as e:
            self.logger.error(f"Secrets scan failed: {e}")
            findings.extend(self._scan_secrets_basic())
        
        return findings

    def _scan_secrets_basic(self) -> List[SecurityFinding]:
        """Basic secret scanning using regex patterns."""
        findings = []
        secret_patterns = [
            (r'(?i)(api[_-]?key|apikey)[\'"\s]*[=:][\'"\s]*[a-zA-Z0-9]{20,}', 'API Key'),
            (r'(?i)(secret[_-]?key|secretkey)[\'"\s]*[=:][\'"\s]*[a-zA-Z0-9]{20,}', 'Secret Key'),
            (r'(?i)(password)[\'"\s]*[=:][\'"\s]*[a-zA-Z0-9]{8,}', 'Password'),
            (r'(?i)(token)[\'"\s]*[=:][\'"\s]*[a-zA-Z0-9]{20,}', 'Token'),
        ]
        
        import re
        
        for file_path in self.project_root.rglob('*.py'):
            if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        for pattern, secret_type in secret_patterns:
                            if re.search(pattern, line):
                                finding = SecurityFinding(
                                    id=f"secret_basic_{secret_type.lower().replace(' ', '_')}_{hash(str(file_path) + str(line_num))}",
                                    title=f"Potential {secret_type}",
                                    description=f"Potential {secret_type.lower()} detected in source code",
                                    severity=Severity.HIGH,
                                    category=Category.SECRET,
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=line_num,
                                    code_snippet=line.strip(),
                                    recommendation=f"Remove or properly secure the {secret_type.lower()}"
                                )
                                findings.append(finding)
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {e}")
        
        return findings

    def scan_dependencies(self) -> List[SecurityFinding]:
        """Scan for dependency-related security issues."""
        findings = []
        
        # Check for outdated dependencies
        try:
            cmd = ['pip', 'list', '--outdated', '--format=json']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout.strip():
                outdated_packages = json.loads(result.stdout)
                
                for package in outdated_packages:
                    finding = SecurityFinding(
                        id=f"outdated_dep_{package['name']}",
                        title=f"Outdated dependency: {package['name']}",
                        description=f"Package {package['name']} is outdated (current: {package['version']}, latest: {package['latest_version']})",
                        severity=Severity.MEDIUM,
                        category=Category.DEPENDENCY,
                        recommendation=f"Update {package['name']} to version {package['latest_version']}"
                    )
                    findings.append(finding)
        except Exception as e:
            self.logger.error(f"Dependency scan failed: {e}")
        
        return findings

    def scan_docker_security(self) -> List[SecurityFinding]:
        """Scan Docker configuration for security issues."""
        findings = []
        
        dockerfile_path = self.project_root / 'Dockerfile'
        if dockerfile_path.exists():
            try:
                with open(dockerfile_path, 'r') as f:
                    dockerfile_content = f.read()
                    
                # Check for common Docker security issues
                security_checks = [
                    ('FROM.*:latest', 'Using latest tag', Severity.MEDIUM, 'Pin specific version tags'),
                    ('USER root', 'Running as root user', Severity.HIGH, 'Use non-root user'),
                    ('ADD.*http', 'Using ADD with URL', Severity.MEDIUM, 'Use COPY instead of ADD for URLs'),
                    ('RUN.*sudo', 'Using sudo in container', Severity.MEDIUM, 'Avoid sudo in containers'),
                ]
                
                import re
                for line_num, line in enumerate(dockerfile_content.split('\n'), 1):
                    for pattern, title, severity, recommendation in security_checks:
                        if re.search(pattern, line, re.IGNORECASE):
                            finding = SecurityFinding(
                                id=f"docker_{title.lower().replace(' ', '_')}_{line_num}",
                                title=f"Docker: {title}",
                                description=f"Docker security issue: {title}",
                                severity=severity,
                                category=Category.CONFIGURATION,
                                file_path='Dockerfile',
                                line_number=line_num,
                                code_snippet=line.strip(),
                                recommendation=recommendation
                            )
                            findings.append(finding)
            except Exception as e:
                self.logger.error(f"Docker security scan failed: {e}")
        
        return findings

    def scan_configuration(self) -> List[SecurityFinding]:
        """Scan configuration files for security issues."""
        findings = []
        
        # Check for common configuration security issues
        config_files = [
            '.env', '.env.example', 'docker-compose.yml', 'docker-compose.yaml',
            'config.py', 'settings.py'
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                findings.extend(self._scan_config_file(file_path))
        
        return findings

    def _scan_config_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan individual configuration file."""
        findings = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for common configuration issues
            import re
            config_checks = [
                (r'DEBUG\s*=\s*True', 'Debug mode enabled', Severity.MEDIUM, 'Disable debug mode in production'),
                (r'SECRET_KEY\s*=\s*[\'"].*[\'"]', 'Hardcoded secret key', Severity.HIGH, 'Use environment variables for secrets'),
                (r'password\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded password', Severity.HIGH, 'Use environment variables for passwords'),
            ]
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern, title, severity, recommendation in config_checks:
                    if re.search(pattern, line, re.IGNORECASE):
                        finding = SecurityFinding(
                            id=f"config_{title.lower().replace(' ', '_')}_{hash(str(file_path) + str(line_num))}",
                            title=f"Configuration: {title}",
                            description=f"Configuration security issue: {title}",
                            severity=severity,
                            category=Category.CONFIGURATION,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation=recommendation
                        )
                        findings.append(finding)
        except Exception as e:
            self.logger.error(f"Configuration scan failed for {file_path}: {e}")
        
        return findings

    def scan_code_quality(self) -> List[SecurityFinding]:
        """Scan for code quality issues that may impact security."""
        findings = []
        
        # This is a placeholder for more advanced code quality checks
        # In a real implementation, this could integrate with tools like:
        # - SonarQube
        # - CodeClimate
        # - Custom static analysis
        
        return findings

    def _generate_summary(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Generate summary statistics from findings."""
        summary = {
            'total': len(findings),
            'critical': sum(1 for f in findings if f.severity == Severity.CRITICAL),
            'high': sum(1 for f in findings if f.severity == Severity.HIGH),
            'medium': sum(1 for f in findings if f.severity == Severity.MEDIUM),
            'low': sum(1 for f in findings if f.severity == Severity.LOW),
            'info': sum(1 for f in findings if f.severity == Severity.INFO),
        }
        
        # Add category breakdown
        for category in Category:
            summary[f'{category.value}_count'] = sum(1 for f in findings if f.category == category)
        
        return summary

    def _count_source_files(self) -> int:
        """Count source files in the project."""
        extensions = ['.py', '.yml', '.yaml', '.json', '.toml', '.cfg', '.ini']
        count = 0
        
        for ext in extensions:
            count += len(list(self.project_root.rglob(f'*{ext}')))
        
        return count


async def main():
    """CLI entry point for security scanning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SafePath Security Scanner')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--output', help='Output file for report')
    parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format')
    parser.add_argument('--severity', choices=['critical', 'high', 'medium', 'low', 'info'], 
                       help='Minimum severity level to report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run security scan
    scanner = SecurityScanner(args.project_root)
    report = scanner.scan_all()
    
    # Filter by severity if specified
    if args.severity:
        severity_order = ['critical', 'high', 'medium', 'low', 'info']
        min_level = severity_order.index(args.severity)
        report.findings = [
            f for f in report.findings 
            if severity_order.index(f.severity.value) <= min_level
        ]
        report.summary = scanner._generate_summary(report.findings)
    
    # Output report
    if args.format == 'json':
        output = report.to_json()
    else:
        # Text format
        output = f"Security Scan Report - {report.timestamp}\n"
        output += f"Scan ID: {report.scan_id}\n"
        output += f"Total Findings: {report.summary['total']}\n\n"
        
        output += "Summary by Severity:\n"
        output += f"  Critical: {report.summary['critical']}\n"
        output += f"  High: {report.summary['high']}\n"
        output += f"  Medium: {report.summary['medium']}\n"
        output += f"  Low: {report.summary['low']}\n"
        output += f"  Info: {report.summary['info']}\n\n"
        
        if report.findings:
            output += "Findings:\n"
            output += "=" * 50 + "\n"
            
            for finding in sorted(report.findings, key=lambda x: x.severity.value):
                output += f"\n[{finding.severity.value.upper()}] {finding.title}\n"
                output += f"Category: {finding.category.value}\n"
                output += f"Description: {finding.description}\n"
                
                if finding.file_path:
                    location = finding.file_path
                    if finding.line_number:
                        location += f":{finding.line_number}"
                    output += f"Location: {location}\n"
                
                if finding.code_snippet:
                    output += f"Code: {finding.code_snippet}\n"
                
                if finding.recommendation:
                    output += f"Recommendation: {finding.recommendation}\n"
                
                output += "-" * 30 + "\n"
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())