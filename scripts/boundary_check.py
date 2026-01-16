#!/usr/bin/env python3
"""
Boundary Check Script - IP Protection for Open Source Repository

Scans files for forbidden patterns that could leak proprietary IP from the
Cleansheet Medical platform into this open source repository.

Usage:
    python scripts/boundary_check.py [path]           # Check specific path
    python scripts/boundary_check.py                  # Check current directory
    python scripts/boundary_check.py --staged         # Check git staged files
    python scripts/boundary_check.py --install-hook   # Install pre-commit hook

Exit codes:
    0 - No violations found
    1 - Violations found
    2 - Script error

Author: Cleansheet LLC
License: CC BY 4.0
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# =============================================================================
# FORBIDDEN PATTERNS - Cleansheet Proprietary IP
# =============================================================================

FORBIDDEN_PATTERNS: dict[str, str] = {
    # Core platform modules
    r"cleansheet-core": "References Cleansheet design system module",
    r"cleansheet-crypto": "References Cleansheet encryption module",
    r"cleansheet-auth": "References Cleansheet authentication module",
    r"cleansheet-sync": "References Cleansheet sync module",
    # UI state management (trade secrets)
    r"canvas-state": "References proprietary UI state management",
    r"canvas-d3-tree": "References proprietary D3 tree component",
    r"canvas-slideouts": "References proprietary slideout system",
    r"canvas-view-preferences": "References proprietary view preferences",
    # Patent-pending innovations
    r"progressive[_\-\s]*disclosure": "References patent-pending Progressive Disclosure UI",
    r"alert[_\-\s]*bubbl": "References patent-pending Alert Bubbling System",
    r"nfc[_\-\s]*proximity": "References patent-pending NFC Proximity System",
    r"clinical[_\-\s]*priority[_\-\s]*algorithm": "References trade secret algorithm",
    # Feature flags and RBAC (trade secrets)
    r"feature-flags\.js": "References proprietary feature flag system",
    r"provider[_\-\s]*persona": "References proprietary persona system",
    r"persona-library": "References proprietary persona definitions",
    r"ui-gating": "References proprietary UI gating system",
    # Storage layer (trade secrets)
    r"encrypted-storage": "References proprietary encrypted storage",
    r"dexie-backend": "References proprietary database backend",
    r"cleansheet-db": "References proprietary database schema",
    r"storage-service": "References proprietary storage service",
    # Medical UI components (copyright)
    r"medical-ui\.js": "References proprietary medical UI module",
    r"medical-badges": "References proprietary badge system",
    r"suggestion-engine": "References proprietary suggestion engine",
    # Platform files
    r"platform-reference-architecture": "References proprietary EHR architecture",
    r"medical\.html": "References proprietary EHR interface",
    r"mockup/": "References proprietary mockup directory",
    # Design system specifics
    r"CleansheetCore\.": "References proprietary CleansheetCore object",
    r"FeatureFlags\.": "References proprietary FeatureFlags object",
    # Internal documentation
    r"intellectual-property-evaluation": "References confidential IP document",
    r"PAUL-AZURE-SECURITY-BLUEPRINT": "References confidential security doc",
}

# Patterns that are warnings but not hard failures
WARNING_PATTERNS: dict[str, str] = {
    r"proprietary": 'Contains word "proprietary" - verify context',
    r"trade\s*secret": 'Contains "trade secret" - verify not exposing details',
    r"patent": 'Contains "patent" - verify not exposing claims',
    r"confidential": 'Contains "confidential" - verify appropriate',
}

# File extensions to check
INCLUDED_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".md",
    ".txt",
    ".rst",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".html",
    ".css",
    ".scss",
    ".sh",
    ".bash",
}

# Files/directories to skip
EXCLUDED_PATHS: set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    "boundary_check.py",
}

# Files allowed to contain certain patterns
ALLOWLIST: dict[str, list[str]] = {
    "LICENSE": ["cleansheet"],
    "README.md": ["cleansheet"],
    "CONTRIBUTING.md": ["cleansheet"],
    "CODE_OF_CONDUCT.md": ["cleansheet"],
    "NOTICE.md": ["cleansheet"],
    "boundary_check.py": ["cleansheet", "proprietary", "patent", "trade secret"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Violation:
    """Represents a single boundary violation."""

    filepath: str
    line_number: int
    pattern: str
    reason: str
    line_content: str
    is_warning: bool = False

    def __str__(self) -> str:
        level = "WARNING" if self.is_warning else "VIOLATION"
        return (
            f"{level}: {self.filepath}:{self.line_number}\n"
            f"  Pattern: {self.pattern}\n"
            f"  Reason: {self.reason}\n"
            f"  Content: {self.line_content.strip()[:80]}..."
        )


@dataclass
class CheckResult:
    """Result of checking a file or directory."""

    files_checked: int
    violations: list[Violation]
    warnings: list[Violation]

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# =============================================================================
# CORE CHECKING LOGIC
# =============================================================================


def should_check_file(filepath: Path) -> bool:
    """Determine if a file should be checked."""
    for excluded in EXCLUDED_PATHS:
        if excluded in filepath.parts:
            return False
    if filepath.suffix.lower() not in INCLUDED_EXTENSIONS:
        return False
    return True


def get_allowlist_patterns(filepath: Path) -> set[str]:
    """Get patterns that are allowed for this specific file."""
    filename = filepath.name
    if filename in ALLOWLIST:
        return set(ALLOWLIST[filename])
    return set()


def check_line(
    line: str, line_number: int, filepath: str, allowed_patterns: set[str]
) -> Iterator[Violation]:
    """Check a single line for violations."""
    # Check forbidden patterns (hard failures)
    for pattern, reason in FORBIDDEN_PATTERNS.items():
        pattern_base = pattern.split("[")[0].split("\\")[0].lower()
        if any(allowed.lower() in pattern_base for allowed in allowed_patterns):
            continue
        if re.search(pattern, line, re.IGNORECASE):
            yield Violation(
                filepath=filepath,
                line_number=line_number,
                pattern=pattern,
                reason=reason,
                line_content=line,
                is_warning=False,
            )

    # Check warning patterns (soft failures)
    for pattern, reason in WARNING_PATTERNS.items():
        pattern_base = pattern.split("[")[0].split("\\")[0].lower()
        if any(allowed.lower() in pattern_base for allowed in allowed_patterns):
            continue
        if re.search(pattern, line, re.IGNORECASE):
            yield Violation(
                filepath=filepath,
                line_number=line_number,
                pattern=pattern,
                reason=reason,
                line_content=line,
                is_warning=True,
            )


def check_file(filepath: Path) -> CheckResult:
    """Check a single file for boundary violations."""
    violations: list[Violation] = []
    warnings: list[Violation] = []

    if not filepath.exists():
        return CheckResult(0, [], [])
    if not should_check_file(filepath):
        return CheckResult(0, [], [])

    allowed_patterns = get_allowlist_patterns(filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, 1):
                for violation in check_line(
                    line, line_number, str(filepath), allowed_patterns
                ):
                    if violation.is_warning:
                        warnings.append(violation)
                    else:
                        violations.append(violation)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return CheckResult(0, [], [])

    return CheckResult(1, violations, warnings)


def check_directory(dirpath: Path, recursive: bool = True) -> CheckResult:
    """Check all files in a directory."""
    all_violations: list[Violation] = []
    all_warnings: list[Violation] = []
    files_checked = 0

    files = dirpath.rglob("*") if recursive else dirpath.glob("*")

    for filepath in files:
        if filepath.is_file():
            result = check_file(filepath)
            files_checked += result.files_checked
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)

    return CheckResult(files_checked, all_violations, all_warnings)


def get_staged_files() -> list[Path]:
    """Get list of git staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [Path(f) for f in result.stdout.strip().split("\n") if f]
    except subprocess.CalledProcessError:
        return []


def check_staged_files() -> CheckResult:
    """Check only git staged files."""
    staged = get_staged_files()
    all_violations: list[Violation] = []
    all_warnings: list[Violation] = []
    files_checked = 0

    for filepath in staged:
        result = check_file(filepath)
        files_checked += result.files_checked
        all_violations.extend(result.violations)
        all_warnings.extend(result.warnings)

    return CheckResult(files_checked, all_violations, all_warnings)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def print_result(result: CheckResult, verbose: bool = False) -> None:
    """Print check results."""
    if result.violations:
        print("\n" + "=" * 70)
        print("BOUNDARY VIOLATIONS DETECTED")
        print("=" * 70)

        by_file: dict[str, list[Violation]] = {}
        for v in result.violations:
            by_file.setdefault(v.filepath, []).append(v)

        for filepath, violations in sorted(by_file.items()):
            print(f"\n{filepath}:")
            for v in violations:
                print(f"  Line {v.line_number}: {v.reason}")
                print(f"    Pattern: {v.pattern}")
                if verbose:
                    print(f"    Content: {v.line_content.strip()[:60]}")

    if result.warnings and (verbose or len(result.warnings) <= 10):
        print("\n" + "-" * 70)
        print("WARNINGS (review manually)")
        print("-" * 70)

        by_file: dict[str, list[Violation]] = {}
        for w in result.warnings:
            by_file.setdefault(w.filepath, []).append(w)

        for filepath, file_warnings in sorted(by_file.items()):
            print(f"\n{filepath}:")
            for w in file_warnings:
                print(f"  Line {w.line_number}: {w.reason}")
    elif result.warnings:
        print(f"\n{len(result.warnings)} warnings suppressed (use --verbose to show)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files checked: {result.files_checked}")
    print(f"Violations: {len(result.violations)}")
    print(f"Warnings: {len(result.warnings)}")

    if result.has_violations:
        print("\nSTATUS: FAILED - Fix violations before committing")
    elif result.has_warnings:
        print("\nSTATUS: PASSED with warnings - Review before release")
    else:
        print("\nSTATUS: PASSED - No IP boundary violations detected")


def print_patterns() -> None:
    """Print all forbidden patterns for reference."""
    print("FORBIDDEN PATTERNS (hard failures):")
    print("-" * 50)
    for pattern, reason in sorted(FORBIDDEN_PATTERNS.items()):
        print(f"  {pattern}")
        print(f"    -> {reason}")

    print("\nWARNING PATTERNS (soft failures):")
    print("-" * 50)
    for pattern, reason in sorted(WARNING_PATTERNS.items()):
        print(f"  {pattern}")
        print(f"    -> {reason}")


# =============================================================================
# PRE-COMMIT HOOK
# =============================================================================

PRE_COMMIT_HOOK = """#!/bin/sh
# Cleansheet IP Boundary Check Pre-commit Hook

echo "Running IP boundary check..."

python scripts/boundary_check.py --staged

if [ $? -ne 0 ]; then
    echo ""
    echo "Commit blocked: IP boundary violations detected"
    echo "Fix violations or use --no-verify to bypass (not recommended)"
    exit 1
fi

exit 0
"""


def install_pre_commit_hook(repo_path: Path) -> bool:
    """Install pre-commit hook in repository."""
    hooks_dir = repo_path / ".git" / "hooks"

    if not hooks_dir.exists():
        print(f"Error: {hooks_dir} does not exist. Is this a git repository?")
        return False

    hook_path = hooks_dir / "pre-commit"

    if hook_path.exists():
        print(f"Warning: {hook_path} already exists")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Aborted")
            return False

    try:
        with open(hook_path, "w") as f:
            f.write(PRE_COMMIT_HOOK)

        # Make executable on Unix systems
        if sys.platform != "win32":
            os.chmod(hook_path, 0o755)

        print(f"Pre-commit hook installed at {hook_path}")
        return True

    except Exception as e:
        print(f"Error installing hook: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for IP boundary violations in open source code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Check current directory
  %(prog)s src/                 Check specific directory
  %(prog)s --staged             Check git staged files only
  %(prog)s --install-hook       Install pre-commit hook
  %(prog)s --list-patterns      Show all forbidden patterns
        """,
    )

    parser.add_argument(
        "path", nargs="?", default=".", help="Path to check (file or directory)"
    )
    parser.add_argument(
        "--staged", action="store_true", help="Check only git staged files"
    )
    parser.add_argument(
        "--pre-commit",
        action="store_true",
        help="Run in pre-commit hook mode (minimal output)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--list-patterns", action="store_true", help="List all forbidden patterns"
    )
    parser.add_argument(
        "--install-hook", action="store_true", help="Install git pre-commit hook"
    )
    parser.add_argument(
        "--warnings-as-errors", action="store_true", help="Treat warnings as errors"
    )

    args = parser.parse_args()

    if args.list_patterns:
        print_patterns()
        return 0

    if args.install_hook:
        success = install_pre_commit_hook(Path("."))
        return 0 if success else 2

    if args.staged or args.pre_commit:
        result = check_staged_files()
    else:
        path = Path(args.path)
        if path.is_file():
            result = check_file(path)
        elif path.is_dir():
            result = check_directory(path)
        else:
            print(f"Error: {path} does not exist")
            return 2

    if not args.pre_commit:
        print_result(result, verbose=args.verbose)
    else:
        if result.has_violations:
            print(f"BLOCKED: {len(result.violations)} IP boundary violations")
            for v in result.violations[:5]:
                print(f"  {v.filepath}:{v.line_number} - {v.reason}")
            if len(result.violations) > 5:
                print(f"  ... and {len(result.violations) - 5} more")

    if result.has_violations:
        return 1
    if args.warnings_as_errors and result.has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
