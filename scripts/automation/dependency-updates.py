#!/usr/bin/env python3
"""
Automated dependency update script for CoT SafePath Filter.
Handles dependency analysis, updates, and security validation.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import toml
from packaging import version


class DependencyUpdater:
    """Automated dependency management and updates."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.security_config = {
            "max_age_days": 30,
            "security_patch_priority": True,
            "auto_update_patch": True,
            "auto_update_minor": False,
            "auto_update_major": False
        }
        
    def analyze_dependencies(self) -> Dict[str, List[Dict]]:
        """Analyze current dependencies for updates and security issues."""
        print("🔍 Analyzing dependencies...")
        
        # Load current dependencies
        with open(self.pyproject_path, 'r') as f:
            pyproject = toml.load(f)
        
        dependencies = pyproject.get('project', {}).get('dependencies', [])
        optional_deps = pyproject.get('project', {}).get('optional-dependencies', {})
        
        results = {
            "outdated": [],
            "security_issues": [],
            "recommendations": []
        }
        
        # Check for outdated packages
        outdated_packages = self._get_outdated_packages()
        for pkg in outdated_packages:
            results["outdated"].append(pkg)
        
        # Check for security vulnerabilities
        security_issues = self._check_security_vulnerabilities()
        for issue in security_issues:
            results["security_issues"].append(issue)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations
        
        return results
    
    def _get_outdated_packages(self) -> List[Dict]:
        """Get list of outdated packages."""
        try:
            # Run pip list --outdated --format=json
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            outdated = json.loads(result.stdout)
            return [
                {
                    "name": pkg["name"],
                    "current_version": pkg["version"],
                    "latest_version": pkg["latest_version"],
                    "type": pkg.get("latest_filetype", "wheel"),
                    "age_days": self._calculate_age_days(pkg["name"], pkg["version"])
                }
                for pkg in outdated
            ]
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error checking outdated packages: {e}")
            return []
    
    def _check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using Safety."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return []  # No vulnerabilities found
            
            # Parse safety output
            vulnerabilities = json.loads(result.stdout)
            return [
                {
                    "package": vuln["package_name"],
                    "installed_version": vuln["installed_version"],
                    "affected_versions": vuln["affected_versions"],
                    "vulnerability_id": vuln["id"],
                    "severity": vuln.get("severity", "unknown"),
                    "description": vuln["advisory"]
                }
                for vuln in vulnerabilities
            ]
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"❌ Error checking security vulnerabilities: {e}")
            return []
    
    def _calculate_age_days(self, package_name: str, version_str: str) -> int:
        """Calculate how many days old a package version is."""
        try:
            # Query PyPI API for package release date
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/{version_str}/json",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                upload_time = data["urls"][0]["upload_time_iso_8601"]
                upload_date = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
                age = datetime.now().astimezone() - upload_date
                return age.days
            
        except Exception as e:
            print(f"⚠️ Could not determine age for {package_name}: {e}")
        
        return 0
    
    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate update recommendations based on analysis."""
        recommendations = []
        
        # Security patches - highest priority
        for issue in analysis["security_issues"]:
            recommendations.append({
                "type": "security",
                "priority": "critical",
                "package": issue["package"],
                "action": "update_immediately",
                "reason": f"Security vulnerability: {issue['vulnerability_id']}",
                "current_version": issue["installed_version"],
                "recommended_version": "latest_safe"
            })
        
        # Outdated packages
        for pkg in analysis["outdated"]:
            current = version.parse(pkg["current_version"])
            latest = version.parse(pkg["latest_version"])
            
            if latest.major > current.major:
                # Major version update
                recommendations.append({
                    "type": "major_update",
                    "priority": "low",
                    "package": pkg["name"],
                    "action": "manual_review",
                    "reason": f"Major version update available ({current} → {latest})",
                    "current_version": str(current),
                    "recommended_version": str(latest)
                })
            elif latest.minor > current.minor:
                # Minor version update
                priority = "medium" if pkg["age_days"] > 30 else "low"
                action = "auto_update" if self.security_config["auto_update_minor"] else "manual_review"
                
                recommendations.append({
                    "type": "minor_update",
                    "priority": priority,
                    "package": pkg["name"],
                    "action": action,
                    "reason": f"Minor version update available ({current} → {latest})",
                    "current_version": str(current),
                    "recommended_version": str(latest)
                })
            else:
                # Patch version update
                action = "auto_update" if self.security_config["auto_update_patch"] else "manual_review"
                
                recommendations.append({
                    "type": "patch_update",
                    "priority": "medium",
                    "package": pkg["name"],
                    "action": action,
                    "reason": f"Patch version update available ({current} → {latest})",
                    "current_version": str(current),
                    "recommended_version": str(latest)
                })
        
        return recommendations
    
    def update_dependencies(self, recommendations: List[Dict], dry_run: bool = True) -> Dict:
        """Execute dependency updates based on recommendations."""
        results = {
            "updated": [],
            "failed": [],
            "skipped": []
        }
        
        for rec in recommendations:
            if rec["action"] == "auto_update" and rec["priority"] in ["critical", "high"]:
                if dry_run:
                    print(f"🔄 [DRY RUN] Would update {rec['package']} to {rec['recommended_version']}")
                    results["updated"].append(rec)
                else:
                    success = self._update_package(rec["package"], rec["recommended_version"])
                    if success:
                        results["updated"].append(rec)
                    else:
                        results["failed"].append(rec)
            else:
                results["skipped"].append(rec)
        
        return results
    
    def _update_package(self, package: str, target_version: str) -> bool:
        """Update a specific package to target version."""
        try:
            if target_version == "latest_safe":
                # For security updates, use latest version
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
            else:
                cmd = [sys.executable, "-m", "pip", "install", f"{package}=={target_version}"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Updated {package} to {target_version}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update {package}: {e}")
            return False
    
    def validate_updates(self) -> bool:
        """Validate that updates don't break the application."""
        print("🧪 Validating updates...")
        
        # Run tests
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print("✅ All tests pass after updates")
                return True
            else:
                print("❌ Tests failed after updates")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def create_update_report(self, analysis: Dict, update_results: Dict) -> str:
        """Create a detailed update report."""
        report = [
            "# Dependency Update Report",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"- **Outdated packages**: {len(analysis['outdated'])}",
            f"- **Security issues**: {len(analysis['security_issues'])}",
            f"- **Updated packages**: {len(update_results['updated'])}",
            f"- **Failed updates**: {len(update_results['failed'])}",
            f"- **Skipped updates**: {len(update_results['skipped'])}",
            "",
        ]
        
        if analysis["security_issues"]:
            report.extend([
                "## 🚨 Security Issues",
                ""
            ])
            for issue in analysis["security_issues"]:
                report.append(f"- **{issue['package']}**: {issue['description']}")
            report.append("")
        
        if update_results["updated"]:
            report.extend([
                "## ✅ Updated Packages",
                ""
            ])
            for update in update_results["updated"]:
                report.append(f"- **{update['package']}**: {update['current_version']} → {update['recommended_version']}")
            report.append("")
        
        if update_results["failed"]:
            report.extend([
                "## ❌ Failed Updates",
                ""
            ])
            for failed in update_results["failed"]:
                report.append(f"- **{failed['package']}**: Failed to update from {failed['current_version']}")
            report.append("")
        
        if analysis["recommendations"]:
            report.extend([
                "## 📋 Manual Review Required",
                ""
            ])
            manual_reviews = [r for r in analysis["recommendations"] if r["action"] == "manual_review"]
            for rec in manual_reviews:
                report.append(f"- **{rec['package']}**: {rec['reason']}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dependency management")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")
    parser.add_argument("--auto-update", action="store_true", help="Automatically apply safe updates")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Initialize updater
    project_root = Path(__file__).parent.parent
    updater = DependencyUpdater(project_root)
    
    print("🚀 Starting dependency analysis...")
    
    # Analyze dependencies
    analysis = updater.analyze_dependencies()
    
    # Perform updates if requested
    if args.report_only:
        update_results = {"updated": [], "failed": [], "skipped": analysis["recommendations"]}
    else:
        dry_run = args.dry_run or not args.auto_update
        update_results = updater.update_dependencies(analysis["recommendations"], dry_run=dry_run)
        
        # Validate updates if not dry run
        if not dry_run and update_results["updated"]:
            if not updater.validate_updates():
                print("❌ Validation failed - consider rolling back updates")
                sys.exit(1)
    
    # Generate report
    report = updater.create_update_report(analysis, update_results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {args.output}")
    else:
        print("\n" + report)
    
    # Exit with appropriate code
    if analysis["security_issues"]:
        print("⚠️ Security issues found - please review and update manually if needed")
        sys.exit(2)
    elif update_results["failed"]:
        print("⚠️ Some updates failed - please review")
        sys.exit(1)
    else:
        print("✅ Dependency analysis complete")
        sys.exit(0)


if __name__ == "__main__":
    main()