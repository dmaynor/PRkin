#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prkin.py - PRkin: PR to Gherkin Consensus Pipeline

LangChain-enabled pipeline:
PR -> patch + context -> hints -> Gemini/Copilot (parallel) ->
Gherkin parse -> weighted consensus -> optional SDLC checks -> CI gate -> artifacts
"""

from __future__ import annotations

# Silence macOS LibreSSL warning from urllib3 (safe: no Python TLS used here)
# Must be before any imports that load urllib3
import warnings
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message=".*urllib3 v2 only supports OpenSSL.*"
)

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from unidiff import PatchSet

from langchain.tools import tool
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
)

# ----------------------------- Constants -----------------------------

SCRIPT_VERSION = "1.5.0"  # Bump on significant changes; used for tracking/schema

DEFAULT_DISAGREEMENT_THRESHOLD = 0.50

HIGH_SIGNAL_TERMS = {
    "must", "should", "shall", "cannot", "can't", "never",
    "regression", "edge", "fail", "error", "timeout",
    "permission", "auth", "token", "upgrade", "downgrade",
}

# ----------------------------- Model Registry -----------------------------

@dataclass
class ModelSpec:
    """Specification for a model CLI backend."""
    name: str
    cli_cmd: List[str]  # Command template, prompt appended or passed via stdin
    prompt_via_stdin: bool  # If True, pass prompt via stdin; else append to cmd
    role: str  # "generation" or "audit"
    strengths: List[str]  # What this model is good at
    timeout: int = 180


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gemini": ModelSpec(
        name="gemini",
        cli_cmd=["gemini", "-p"],
        prompt_via_stdin=False,
        role="generation",
        strengths=["api", "reasoning", "edge cases"],
        timeout=180,
    ),
    "copilot": ModelSpec(
        name="copilot",
        cli_cmd=["copilot", "-p"],
        prompt_via_stdin=False,
        role="generation",
        strengths=["code patterns", "refactoring", "infrastructure"],
        timeout=180,
    ),
    "codex": ModelSpec(
        name="codex",
        cli_cmd=["codex", "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"],
        prompt_via_stdin=True,
        role="generation",
        strengths=["structural tests", "api surfaces", "build tooling"],
        timeout=300,  # Codex can be slower
    ),
    "claude": ModelSpec(
        name="claude",
        cli_cmd=["claude", "-p", "--output-format", "text"],
        prompt_via_stdin=False,
        role="audit",
        strengths=["reasoning", "auditing", "evidence validation"],
        timeout=240,  # Claude can be thorough
    ),
}

# Default model sets
DEFAULT_GENERATION_MODELS = ["copilot"]
DEFAULT_AUDIT_MODEL = "claude"
DEFAULT_PARALLEL_WORKERS = 2

# Presets for common configurations (based on telemetry analysis)
PRESETS = {
    "fast": {
        "description": "Single agent, fastest execution (copilot)",
        "models": ["copilot"],
        "threshold": 0.50,
        "parallel": 2,
        "audit": False,
    },
    "balanced": {
        "description": "Two agents with complementary coverage (copilot + codex)",
        "models": ["copilot", "codex"],
        "threshold": 0.55,
        "parallel": 2,
        "audit": False,
    },
    "thorough": {
        "description": "Two agents + audit for hallucination detection",
        "models": ["copilot", "codex"],
        "threshold": 0.65,
        "parallel": 2,
        "audit": True,
        "audit_model": "claude",
    },
}

# Domain weights extended for new models
DOMAIN_WEIGHTS = {
    "api": {"gemini": 0.80, "copilot": 0.55, "codex": 0.70, "claude": 0.75},
    "infra": {"gemini": 0.55, "copilot": 0.75, "codex": 0.65, "claude": 0.60},
    "crypto": {"gemini": 0.70, "copilot": 0.50, "codex": 0.55, "claude": 0.80},
    "default": {"gemini": 0.65, "copilot": 0.65, "codex": 0.60, "claude": 0.70},
}


def get_model_spec(model: str) -> ModelSpec:
    """Get model specification, raising error if unknown."""
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model]


def check_binary_available(cmd: str) -> bool:
    """
    Check if a binary/command is available on PATH.
    Uses shutil.which() for reliable cross-platform detection.
    """
    return shutil.which(cmd) is not None


def check_model_available(model: str) -> bool:
    """
    Check if a model's CLI tool is installed and available.
    Returns True if the model can be invoked, False otherwise.
    """
    if model not in MODEL_REGISTRY:
        return False

    spec = MODEL_REGISTRY[model]
    base_cmd = spec.cli_cmd[0]
    return check_binary_available(base_cmd)


# ----------------------------- Authorization Checks -----------------------------

def check_gemini_auth() -> bool:
    """
    Check if Gemini CLI is authenticated.
    Gemini uses web-based OAuth; cached credentials show "Loaded cached credentials."
    """
    if not check_binary_available("gemini"):
        return False
    try:
        # Run a minimal prompt to check if auth works
        # The -p flag runs non-interactively and exits
        result = subprocess.run(
            ["gemini", "-p", "reply with only: ok"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        combined_lower = combined.lower()

        # Check for cached credentials - this means auth is valid
        if "loaded cached credentials" in combined_lower:
            return True

        # Check for successful response
        if result.returncode == 0:
            return True

        # Check for explicit auth-related errors (not quota/rate limits)
        if "not authenticated" in combined_lower or "login required" in combined_lower:
            return False

        # Quota errors or other API errors mean auth IS working
        if "quota" in combined_lower or "rate limit" in combined_lower:
            return True

        # If it ran without auth errors, consider it authenticated
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Timeout likely means it's trying to run (authenticated)
        return True
    except Exception:
        return False


def check_copilot_auth() -> bool:
    """
    Check if Copilot CLI is authenticated via GitHub.
    Copilot auth is tied to GitHub authentication.
    """
    if not check_binary_available("gh"):
        return False
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_codex_auth() -> bool:
    """
    Check if Codex CLI is authenticated.
    Codex supports both OPENAI_API_KEY and web-based OAuth authentication.
    When not authenticated, codex exec returns 401 Unauthorized errors.
    """
    if not check_binary_available("codex"):
        return False

    # Check for API key in environment (one auth method)
    if "OPENAI_API_KEY" in os.environ:
        return True

    # For web-based auth, run a minimal command to verify
    # Using exec with a simple prompt to test authentication
    try:
        result = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "echo authenticated"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Check for auth-related errors in output
        stderr = result.stderr.lower() if result.stderr else ""
        stdout = result.stdout.lower() if result.stdout else ""
        combined = stderr + stdout

        # Codex shows "401 Unauthorized" or "Missing bearer" when not authenticated
        if "401" in combined or "unauthorized" in combined or "missing bearer" in combined:
            return False
        if "not authenticated" in combined or "login required" in combined:
            return False

        # If command succeeded without auth errors, we're authenticated
        if result.returncode == 0:
            return True

        # Non-zero exit without auth errors could be other issues
        # Check if it at least started (shows session info)
        if "session id:" in combined or "model:" in combined:
            # It started but failed for non-auth reasons - still authenticated
            return "401" not in combined and "unauthorized" not in combined

        return False
    except subprocess.TimeoutExpired:
        # Timeout after starting likely means it's running (authenticated)
        return True
    except Exception:
        return False


def check_claude_auth() -> bool:
    """
    Check if Claude CLI is authenticated.
    Claude Code uses web-based OAuth; we verify by running a minimal prompt.
    """
    if not check_binary_available("claude"):
        return False
    try:
        # Run a minimal prompt to verify auth works
        # -p flag runs non-interactively with the prompt as argument
        result = subprocess.run(
            ["claude", "-p", "reply with only: ok", "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True
        # Check for auth-related errors
        stderr = result.stderr.lower() if result.stderr else ""
        stdout = result.stdout.lower() if result.stdout else ""
        combined = stderr + stdout
        if "not authenticated" in combined or "login" in combined or "unauthorized" in combined:
            return False
        # If it ran without auth errors, might still be authenticated
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # Timeout likely means it's trying to run (authenticated)
        return True
    except Exception:
        return False


def check_agent_auth(model: str) -> bool:
    """
    Check authorization status for a specific agent/model.
    Returns True if the agent is authenticated, False otherwise.
    """
    auth_checks = {
        "gemini": check_gemini_auth,
        "copilot": check_copilot_auth,
        "codex": check_codex_auth,
        "claude": check_claude_auth,
    }

    if model not in auth_checks:
        # Unknown model - assume auth OK if binary exists
        return check_model_available(model)

    return auth_checks[model]()


# ----------------------------- Model Identity -----------------------------

def get_model_info(model: str) -> Dict[str, str]:
    """
    Get model identity information including version string.
    Returns dict with 'binary_path', 'version_string', 'reported_model'.
    """
    if model not in MODEL_REGISTRY:
        return {
            "binary_path": "unknown",
            "version_string": "unknown",
            "reported_model": "unknown",
        }

    spec = MODEL_REGISTRY[model]
    base_cmd = spec.cli_cmd[0]

    # Get binary path
    binary_path = shutil.which(base_cmd) or "not found"

    # Get version string
    version_string = "unknown"
    try:
        result = subprocess.run(
            [base_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            version_string = result.stdout.strip().split('\n')[0]  # First line only
        elif result.stderr.strip():
            version_string = result.stderr.strip().split('\n')[0]
    except Exception:
        pass

    # If --version didn't work, try -v or version subcommand
    if version_string == "unknown":
        for variant in [[base_cmd, "-v"], [base_cmd, "version"], [base_cmd, "-V"]]:
            try:
                result = subprocess.run(
                    variant,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    version_string = result.stdout.strip().split('\n')[0]
                    break
            except Exception:
                continue

    return {
        "binary_path": binary_path,
        "version_string": version_string,
        "reported_model": f"{model} ({version_string})" if version_string != "unknown" else model,
    }


@dataclass
class AgentValidationResult:
    """Result of validating an agent's capability to participate."""
    model: str
    binary_available: bool
    binary_path: str
    auth_status: bool
    auth_verified: bool  # True if auth was actually checked, False if assumed
    version_string: str
    reported_model: str
    can_participate: bool
    failure_reason: Optional[str] = None


def validate_agent(model: str) -> AgentValidationResult:
    """
    Perform full validation of an agent: binary, auth, and identity.

    An agent may only participate if:
    - Binary is available (can it run?)
    - Agent is authenticated (is it authorized?)
    - Agent declares its model identity (what is it?)

    Returns AgentValidationResult with all validation data.
    """
    # Check 1: Binary availability
    binary_available = check_model_available(model)
    if not binary_available:
        return AgentValidationResult(
            model=model,
            binary_available=False,
            binary_path="not found",
            auth_status=False,
            auth_verified=True,  # We did check - binary not found
            version_string="unknown",
            reported_model="unknown",
            can_participate=False,
            failure_reason=f"Agent '{model}' is not installed or not on PATH.",
        )

    # Check 2: Authorization
    auth_status = check_agent_auth(model)
    if not auth_status:
        model_info = get_model_info(model)
        return AgentValidationResult(
            model=model,
            binary_available=True,
            binary_path=model_info["binary_path"],
            auth_status=False,
            auth_verified=True,  # We actually checked auth
            version_string=model_info["version_string"],
            reported_model=model_info["reported_model"],
            can_participate=False,
            failure_reason=f"Agent '{model}' is not authenticated. Please authenticate first.",
        )

    # Check 3: Model identity
    model_info = get_model_info(model)

    return AgentValidationResult(
        model=model,
        binary_available=True,
        binary_path=model_info["binary_path"],
        auth_status=True,
        auth_verified=True,  # We actually checked auth
        version_string=model_info["version_string"],
        reported_model=model_info["reported_model"],
        can_participate=True,
        failure_reason=None,
    )


def validate_all_agents(models: List[str]) -> Dict[str, AgentValidationResult]:
    """
    Validate all specified agents and return results.
    """
    return {model: validate_agent(model) for model in models}


def discover_available_models() -> Dict[str, bool]:
    """
    Discover which models are available on this system.
    Returns a dict of model_name -> is_available.
    """
    available = {}
    for model in MODEL_REGISTRY:
        available[model] = check_model_available(model)
    return available


def get_available_models() -> List[str]:
    """Get list of models that are available on this system."""
    return [m for m, available in discover_available_models().items() if available]


def validate_models(models: List[str], check_availability: bool = True) -> List[str]:
    """
    Validate that all specified models exist in registry and are available.
    Returns list of unavailable models (empty if all OK).
    """
    unavailable = []
    for model in models:
        if model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}")
        if check_availability and not check_model_available(model):
            unavailable.append(model)
    return unavailable


def print_model_status() -> None:
    """Print comprehensive status of all registered models including auth and version."""
    print("\nAgent Capability Status:")
    print("=" * 70)
    print(f"{'Agent':<12} {'Role':<12} {'Binary':<10} {'Auth':<10} {'Version'}")
    print("-" * 70)

    authorized_count = 0
    for model, spec in MODEL_REGISTRY.items():
        validation = validate_agent(model)

        binary_status = "✓" if validation.binary_available else "✗"
        auth_status = "✓" if validation.auth_status else "✗"
        version = validation.version_string[:25] if validation.version_string else "unknown"

        if validation.can_participate:
            authorized_count += 1
            print(f"  {model:<10} {spec.role:<12} {binary_status:<10} {auth_status:<10} {version}")
        else:
            print(f"  {model:<10} {spec.role:<12} {binary_status:<10} {auth_status:<10} {version}")
            if validation.failure_reason:
                print(f"    └─ {validation.failure_reason}")

    print("-" * 70)
    print(f"  {authorized_count}/{len(MODEL_REGISTRY)} agents ready to participate\n")


# ----------------------------- Telemetry Database -----------------------------

class TelemetryDB:
    """
    SQLite-based telemetry database for tracking pipeline runs and results.

    This is a curated historical database for learning from results over time,
    separate from the output directory artifacts.

    Schema design goals:
    - Track full workflow: which models generated, which audited
    - Store raw Gherkin outputs for each model
    - Record confidence scores and classifications
    - Enable queries like "which model is best for API layer scenarios?"
    """

    # Schema version - increment when making breaking changes
    SCHEMA_VERSION = 5  # v5: Added resolved_config, run_classification, dispute_reason, auth_verified

    SCHEMA = """
    -- Schema metadata table for versioning
    CREATE TABLE IF NOT EXISTS schema_info (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Runs table: each execution of the pipeline
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        repo TEXT NOT NULL,
        pr_number INTEGER NOT NULL,
        threshold REAL NOT NULL,
        audit_enabled INTEGER NOT NULL,
        generation_models TEXT NOT NULL,  -- JSON array
        audit_model TEXT,
        output_dir TEXT NOT NULL,
        reasoning_file TEXT,  -- Path to reasoning transcript log
        files_total INTEGER DEFAULT 0,
        files_succeeded INTEGER DEFAULT 0,
        files_failed INTEGER DEFAULT 0,
        files_disputed INTEGER DEFAULT 0,
        duration_seconds REAL,
        script_version TEXT,  -- Script version for tracking
        preset_name TEXT,  -- Preset used (fast, balanced, thorough, or NULL)
        resolved_config TEXT,  -- JSON blob of fully resolved configuration
        run_classification TEXT  -- baseline, coverage_exploration, audit_strict, mixed
    );

    -- File results: each file processed in a run
    CREATE TABLE IF NOT EXISTS file_results (
        file_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        file_path TEXT NOT NULL,
        status TEXT NOT NULL,  -- 'success', 'disputed', 'failed'
        failure_type TEXT,
        error_message TEXT,
        disagreement REAL,
        dispute_reason TEXT,  -- coverage_diversity, contradiction, mixed (when status=disputed)
        domain TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    -- Model outputs: raw Gherkin from each model
    CREATE TABLE IF NOT EXISTS model_outputs (
        output_id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        role TEXT NOT NULL,  -- 'generation', 'audit'
        raw_output TEXT,
        cleaned_output TEXT,
        parse_success INTEGER,
        failure_type TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (file_id) REFERENCES file_results(file_id)
    );

    -- Scenarios: each scenario in the merged output
    CREATE TABLE IF NOT EXISTS scenarios (
        scenario_id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        scenario_name TEXT NOT NULL,
        steps_json TEXT NOT NULL,  -- JSON array of steps
        tags_json TEXT,  -- JSON array of tags
        classification TEXT,  -- grounded, inferred, contextual, speculative, hallucinated
        confidence REAL,
        layer TEXT,  -- api, implementation, user-observable, unknown
        evidence_symbols_json TEXT,  -- JSON array
        FOREIGN KEY (file_id) REFERENCES file_results(file_id)
    );

    -- Scenario sources: which models contributed to each scenario
    CREATE TABLE IF NOT EXISTS scenario_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scenario_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        FOREIGN KEY (scenario_id) REFERENCES scenarios(scenario_id)
    );

    -- Audit findings: findings from audit
    CREATE TABLE IF NOT EXISTS audit_findings (
        finding_id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        scenario_name TEXT NOT NULL,
        claim TEXT NOT NULL,
        auditor_model TEXT NOT NULL,
        severity TEXT,
        FOREIGN KEY (file_id) REFERENCES file_results(file_id)
    );

    -- Model performance: per-model metrics per file
    CREATE TABLE IF NOT EXISTS model_performance (
        perf_id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        role TEXT NOT NULL,  -- 'generation', 'audit'
        scenarios_produced INTEGER,
        avg_confidence REAL,
        grounded_claims INTEGER,
        grounded_ratio REAL,
        findings_count INTEGER,
        hallucinations_flagged INTEGER,
        scenarios_reviewed INTEGER,
        FOREIGN KEY (file_id) REFERENCES file_results(file_id),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    -- Reasoning messages: explanations emitted at decision boundaries
    CREATE TABLE IF NOT EXISTS reasoning_messages (
        message_id TEXT PRIMARY KEY,
        run_id TEXT,
        timestamp TEXT NOT NULL,
        role TEXT NOT NULL,  -- 'auditor', 'generator', 'system'
        model TEXT NOT NULL,  -- 'gemini', 'copilot', 'claude', 'system'
        type TEXT NOT NULL,  -- 'CLAIM', 'EVIDENCE', 'CHALLENGE', 'DECISION'
        content TEXT NOT NULL,  -- Plain English explanation
        file TEXT,
        scenario TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    -- Agent validations: tracks agent capability checks at startup
    -- Records binary availability, auth status, and model identity per run
    CREATE TABLE IF NOT EXISTS agent_validations (
        validation_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        binary_available INTEGER NOT NULL,  -- 0 or 1
        binary_path TEXT,
        auth_status INTEGER NOT NULL,  -- 0 or 1
        auth_verified INTEGER NOT NULL DEFAULT 0,  -- 0=assumed, 1=actually verified
        version_string TEXT,
        reported_model TEXT,
        can_participate INTEGER NOT NULL,  -- 0 or 1
        failure_reason TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_file_results_run ON file_results(run_id);
    CREATE INDEX IF NOT EXISTS idx_model_outputs_file ON model_outputs(file_id);
    CREATE INDEX IF NOT EXISTS idx_scenarios_file ON scenarios(file_id);
    CREATE INDEX IF NOT EXISTS idx_scenarios_classification ON scenarios(classification);
    CREATE INDEX IF NOT EXISTS idx_scenarios_layer ON scenarios(layer);
    CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name);
    CREATE INDEX IF NOT EXISTS idx_model_performance_run ON model_performance(run_id);
    CREATE INDEX IF NOT EXISTS idx_runs_repo_pr ON runs(repo, pr_number);
    CREATE INDEX IF NOT EXISTS idx_reasoning_run ON reasoning_messages(run_id);
    CREATE INDEX IF NOT EXISTS idx_reasoning_type ON reasoning_messages(type);
    CREATE INDEX IF NOT EXISTS idx_reasoning_file ON reasoning_messages(file);
    CREATE INDEX IF NOT EXISTS idx_agent_validations_run ON agent_validations(run_id);
    CREATE INDEX IF NOT EXISTS idx_agent_validations_agent ON agent_validations(agent_name);
    """

    def __init__(self, db_path: str):
        """Initialize database connection and create schema if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        self._check_schema_version()

    def _create_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def _check_schema_version(self) -> None:
        """Check and update schema version, apply migrations if needed."""
        try:
            cursor = self.conn.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            )
            row = cursor.fetchone()
            if row:
                db_version = int(row[0])
                if db_version < self.SCHEMA_VERSION:
                    sys.stderr.write(
                        f"[info] Migrating database schema from v{db_version} to v{self.SCHEMA_VERSION}...\n"
                    )
                    self._apply_migrations(db_version)
                    self._set_schema_version()
                    sys.stderr.write(f"[info] Database migration complete.\n")
                elif db_version > self.SCHEMA_VERSION:
                    sys.stderr.write(
                        f"[warn] Database schema version {db_version} is newer than "
                        f"script version {self.SCHEMA_VERSION}. Consider updating the script.\n"
                    )
            else:
                # No version set, initialize it
                self._set_schema_version()
        except sqlite3.OperationalError:
            # Table doesn't exist yet, will be created
            self._set_schema_version()

    def _apply_migrations(self, from_version: int) -> None:
        """Apply incremental migrations from from_version to SCHEMA_VERSION."""
        # Migration from v2 to v3: Add reasoning_file column to runs
        if from_version < 3:
            try:
                self.conn.execute("ALTER TABLE runs ADD COLUMN reasoning_file TEXT")
                self.conn.commit()
                sys.stderr.write("[info] Added reasoning_file column to runs table.\n")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    pass  # Column already exists
                else:
                    raise

        # Migration from v3 to v4: Add agent_validations table
        if from_version < 4:
            try:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_validations (
                        validation_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        binary_available INTEGER NOT NULL,
                        binary_path TEXT,
                        auth_status INTEGER NOT NULL,
                        version_string TEXT,
                        reported_model TEXT,
                        can_participate INTEGER NOT NULL,
                        failure_reason TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    )
                """)
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_validations_run ON agent_validations(run_id)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_validations_agent ON agent_validations(agent_name)"
                )
                self.conn.commit()
                sys.stderr.write("[info] Added agent_validations table.\n")
            except sqlite3.OperationalError as e:
                if "already exists" in str(e).lower():
                    pass  # Table already exists
                else:
                    raise

        # Migration from v4 to v5: Add new tracking columns
        if from_version < 5:
            # Add columns to runs table
            for col_def in [
                ("files_disputed", "INTEGER DEFAULT 0"),
                ("script_version", "TEXT"),
                ("preset_name", "TEXT"),
                ("resolved_config", "TEXT"),
                ("run_classification", "TEXT"),
            ]:
                try:
                    self.conn.execute(f"ALTER TABLE runs ADD COLUMN {col_def[0]} {col_def[1]}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise

            # Add dispute_reason to file_results
            try:
                self.conn.execute("ALTER TABLE file_results ADD COLUMN dispute_reason TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            # Add auth_verified to agent_validations
            try:
                self.conn.execute(
                    "ALTER TABLE agent_validations ADD COLUMN auth_verified INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

            self.conn.commit()
            sys.stderr.write("[info] Migrated schema to v5: added tracking columns.\n")

    def _set_schema_version(self) -> None:
        """Set the current schema version in the database."""
        self.conn.execute("""
            INSERT OR REPLACE INTO schema_info (key, value, updated_at)
            VALUES ('version', ?, ?)
        """, (str(self.SCHEMA_VERSION), datetime.now().isoformat()))
        self.conn.execute("""
            INSERT OR REPLACE INTO schema_info (key, value, updated_at)
            VALUES ('script_name', ?, ?)
        """, ('prkin.py', datetime.now().isoformat()))
        self.conn.commit()

    def get_schema_version(self) -> Optional[int]:
        """Get the current schema version from the database."""
        try:
            cursor = self.conn.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            )
            row = cursor.fetchone()
            return int(row[0]) if row else None
        except sqlite3.OperationalError:
            return None

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def start_run(self, repo: str, pr_number: int, threshold: float,
                  audit_enabled: bool, generation_models: List[str],
                  audit_model: Optional[str], output_dir: str,
                  reasoning_file: Optional[str] = None,
                  script_version: Optional[str] = None,
                  preset_name: Optional[str] = None,
                  resolved_config: Optional[Dict[str, Any]] = None,
                  run_classification: Optional[str] = None) -> str:
        """Record the start of a pipeline run. Returns run_id."""
        run_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO runs (run_id, timestamp, repo, pr_number, threshold,
                             audit_enabled, generation_models, audit_model, output_dir,
                             reasoning_file, script_version, preset_name,
                             resolved_config, run_classification)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now().isoformat(),
            repo,
            pr_number,
            threshold,
            1 if audit_enabled else 0,
            json.dumps(generation_models),
            audit_model,
            output_dir,
            reasoning_file,
            script_version,
            preset_name,
            json.dumps(resolved_config) if resolved_config else None,
            run_classification,
        ))
        self.conn.commit()
        return run_id

    def finish_run(self, run_id: str, files_total: int, files_succeeded: int,
                   files_failed: int, duration_seconds: float,
                   files_disputed: int = 0) -> None:
        """Record the completion of a pipeline run."""
        self.conn.execute("""
            UPDATE runs SET files_total = ?, files_succeeded = ?,
                           files_failed = ?, files_disputed = ?, duration_seconds = ?
            WHERE run_id = ?
        """, (files_total, files_succeeded, files_failed, files_disputed,
              duration_seconds, run_id))
        self.conn.commit()

    def log_agent_validation(self, run_id: str, validation: 'AgentValidationResult') -> str:
        """
        Log agent validation result for auditability.
        Records binary availability, auth status, and model identity.
        Returns validation_id.
        """
        validation_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO agent_validations (
                validation_id, run_id, agent_name, binary_available, binary_path,
                auth_status, auth_verified, version_string, reported_model, can_participate,
                failure_reason, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            validation_id,
            run_id,
            validation.model,
            1 if validation.binary_available else 0,
            validation.binary_path,
            1 if validation.auth_status else 0,
            1 if validation.auth_verified else 0,
            validation.version_string,
            validation.reported_model,
            1 if validation.can_participate else 0,
            validation.failure_reason,
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        return validation_id

    def get_agent_validations(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all agent validations for a run."""
        cursor = self.conn.execute("""
            SELECT * FROM agent_validations WHERE run_id = ?
            ORDER BY agent_name
        """, (run_id,))
        return [dict(row) for row in cursor.fetchall()]

    def log_file_result(self, run_id: str, file_path: str, status: str,
                        failure_type: Optional[str] = None,
                        error_message: Optional[str] = None,
                        disagreement: Optional[float] = None,
                        dispute_reason: Optional[str] = None,
                        domain: Optional[str] = None) -> str:
        """Log a file processing result. Returns file_id."""
        file_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO file_results (file_id, run_id, file_path, status,
                                      failure_type, error_message, disagreement,
                                      dispute_reason, domain, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_id,
            run_id,
            file_path,
            status,
            failure_type,
            error_message,
            disagreement,
            dispute_reason,
            domain,
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        return file_id

    def log_model_output(self, file_id: str, model_name: str, role: str,
                         raw_output: str, cleaned_output: Optional[str] = None,
                         parse_success: bool = True,
                         failure_type: Optional[str] = None) -> str:
        """Log raw model output. Returns output_id."""
        output_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO model_outputs (output_id, file_id, model_name, role,
                                       raw_output, cleaned_output, parse_success,
                                       failure_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            output_id,
            file_id,
            model_name,
            role,
            raw_output,
            cleaned_output,
            1 if parse_success else 0,
            failure_type,
            datetime.now().isoformat(),
        ))
        self.conn.commit()
        return output_id

    def log_scenario(self, file_id: str, scenario_name: str, steps: List[str],
                     tags: List[str], classification: str, confidence: float,
                     layer: str, evidence_symbols: List[str],
                     generating_models: List[str]) -> str:
        """Log a merged scenario with its classification. Returns scenario_id."""
        scenario_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO scenarios (scenario_id, file_id, scenario_name, steps_json,
                                   tags_json, classification, confidence, layer,
                                   evidence_symbols_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scenario_id,
            file_id,
            scenario_name,
            json.dumps(steps),
            json.dumps(tags),
            classification,
            confidence,
            layer,
            json.dumps(evidence_symbols),
        ))

        # Log which models contributed to this scenario
        for model in generating_models:
            self.conn.execute("""
                INSERT INTO scenario_sources (scenario_id, model_name)
                VALUES (?, ?)
            """, (scenario_id, model))

        self.conn.commit()
        return scenario_id

    def log_audit_finding(self, file_id: str, scenario_name: str, claim: str,
                          auditor_model: str, severity: Optional[str] = None) -> str:
        """Log an audit finding. Returns finding_id."""
        finding_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO audit_findings (finding_id, file_id, scenario_name,
                                        claim, auditor_model, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (finding_id, file_id, scenario_name, claim, auditor_model, severity))
        self.conn.commit()
        return finding_id

    def log_model_performance(self, file_id: str, run_id: str, model_name: str,
                              role: str, metrics: Dict[str, Any]) -> str:
        """Log model performance metrics. Returns perf_id."""
        perf_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO model_performance (perf_id, file_id, run_id, model_name, role,
                                           scenarios_produced, avg_confidence,
                                           grounded_claims, grounded_ratio,
                                           findings_count, hallucinations_flagged,
                                           scenarios_reviewed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            perf_id,
            file_id,
            run_id,
            model_name,
            role,
            metrics.get("scenarios_produced"),
            metrics.get("avg_confidence"),
            metrics.get("grounded_claims"),
            metrics.get("grounded_ratio"),
            metrics.get("findings_count"),
            metrics.get("hallucinations_flagged"),
            metrics.get("scenarios_reviewed"),
        ))
        self.conn.commit()
        return perf_id

    # -------------------- Query Methods for Analysis --------------------

    def get_model_stats(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        Get aggregated statistics for models.
        If model_name is provided, filter to that model.
        """
        query = """
            SELECT
                model_name,
                role,
                COUNT(*) as total_files,
                AVG(avg_confidence) as overall_avg_confidence,
                SUM(grounded_claims) as total_grounded,
                AVG(grounded_ratio) as avg_grounded_ratio,
                SUM(scenarios_produced) as total_scenarios,
                SUM(findings_count) as total_findings,
                SUM(hallucinations_flagged) as total_hallucinations
            FROM model_performance
        """
        params = []
        if model_name:
            query += " WHERE model_name = ?"
            params.append(model_name)
        query += " GROUP BY model_name, role"

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_layer_stats(self) -> List[Dict]:
        """Get statistics by layer (api, implementation, user-observable)."""
        cursor = self.conn.execute("""
            SELECT
                layer,
                classification,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM scenarios
            GROUP BY layer, classification
            ORDER BY layer, classification
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_run_history(self, repo: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get recent pipeline runs."""
        query = """
            SELECT run_id, timestamp, repo, pr_number, files_total,
                   files_succeeded, files_failed, duration_seconds,
                   generation_models, audit_model
            FROM runs
        """
        params = []
        if repo:
            query += " WHERE repo = ?"
            params.append(repo)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_hallucination_rate_by_model(self) -> List[Dict]:
        """Get hallucination rates by generating model."""
        cursor = self.conn.execute("""
            SELECT
                ss.model_name,
                COUNT(DISTINCT s.scenario_id) as total_scenarios,
                COUNT(DISTINCT af.finding_id) as flagged_scenarios,
                CAST(COUNT(DISTINCT af.finding_id) AS REAL) /
                    NULLIF(COUNT(DISTINCT s.scenario_id), 0) as hallucination_rate
            FROM scenario_sources ss
            JOIN scenarios s ON ss.scenario_id = s.scenario_id
            LEFT JOIN audit_findings af ON s.scenario_name = af.scenario_name
                AND s.file_id = af.file_id
            GROUP BY ss.model_name
        """)
        return [dict(row) for row in cursor.fetchall()]


# Global telemetry instance (set in main)
_telemetry_db: Optional[TelemetryDB] = None


def get_telemetry_db() -> Optional[TelemetryDB]:
    """Get the global telemetry database instance."""
    return _telemetry_db


# ----------------------------- Reasoning Bus -----------------------------

class MessageType:
    """Fixed set of reasoning message types."""
    CLAIM = "CLAIM"           # A model asserts a behavior or invariant
    EVIDENCE = "EVIDENCE"     # A reference to concrete artifacts supporting a claim
    CHALLENGE = "CHALLENGE"   # Cannot validate a claim from available evidence
    DECISION = "DECISION"     # System explains what it decided and why (system-only)


@dataclass
class ReasoningMessage:
    """
    A single reasoning message emitted at a decision boundary.

    Messages are natural language explanations of decisions that already happened.
    They never trigger actions, mutate artifacts, or influence execution.
    """
    timestamp: str
    role: str       # "auditor", "generator", "system"
    model: str      # "gemini", "copilot", "claude", "system"
    msg_type: str   # MessageType value
    content: str    # Plain English explanation
    file: Optional[str] = None
    scenario: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "role": self.role,
            "model": self.model,
            "type": self.msg_type,
            "content": self.content,
            "file": self.file,
            "scenario": self.scenario,
            "run_id": self.run_id,
        }

    def to_line(self) -> str:
        """Format as human-readable single line."""
        # Format: [timestamp] [TYPE] [model] [role] [file] [scenario]: content
        prefix = f"[{self.timestamp[:19]}] [{self.msg_type}] [{self.model}] [{self.role}]"
        location = ""
        if self.file:
            location = f" [{self.file}]"
        if self.scenario:
            location += f" [{self.scenario}]"
        return f"{prefix}{location}: {self.content}"


class ReasoningBus:
    """
    Append-only, inspectable bus that emits natural-language explanations
    of system decisions at key points in the pipeline.

    Core principle: Decisions are made by deterministic gates.
                   Explanations are emitted as natural language.
                   Adjustments are made only by humans.

    This loop exists to inform, not to control.

    Hard constraints:
    - No code in messages
    - No commands
    - No suggestions that imply action
    - No "fix", "change", or "apply" language
    """

    # Words that imply action - messages containing these should be flagged
    ACTION_WORDS = frozenset([
        "fix", "change", "apply", "modify", "update", "edit", "replace",
        "remove", "add", "insert", "delete", "should", "must", "need to",
        "recommend", "suggest", "try", "consider changing"
    ])

    def __init__(self, run_id: Optional[str] = None, output_file: Optional[str] = None):
        """
        Initialize reasoning bus.

        Args:
            run_id: Optional run ID for correlation with telemetry
            output_file: Optional file path to write messages (appends)
        """
        self.run_id = run_id
        self.output_file = output_file
        self.messages: List[ReasoningMessage] = []
        self._file_handle = None

        if output_file:
            self._file_handle = open(output_file, "a", encoding="utf-8")

    def close(self) -> None:
        """Close output file if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _validate_content(self, content: str) -> str:
        """
        Validate content doesn't imply action.
        If it does, append human review notice.
        """
        content_lower = content.lower()
        for word in self.ACTION_WORDS:
            if word in content_lower:
                return f"{content} [This requires human review.]"
        return content

    def _emit(self, role: str, model: str, msg_type: str, content: str,
              file: Optional[str] = None, scenario: Optional[str] = None) -> ReasoningMessage:
        """Internal emit method."""
        # Validate content doesn't imply action
        validated_content = self._validate_content(content)

        msg = ReasoningMessage(
            timestamp=datetime.now().isoformat(),
            role=role,
            model=model,
            msg_type=msg_type,
            content=validated_content,
            file=file,
            scenario=scenario,
            run_id=self.run_id,
        )

        self.messages.append(msg)

        # Write to file if configured
        if self._file_handle:
            self._file_handle.write(msg.to_line() + "\n")
            self._file_handle.flush()

        return msg

    # -------------------- Public Emission Methods --------------------

    def claim(self, model: str, content: str, file: Optional[str] = None,
              scenario: Optional[str] = None) -> ReasoningMessage:
        """
        Emit a CLAIM: A model asserts a behavior or invariant.

        Example: "Scans succeed when OS metadata is missing."
        """
        role = "generator" if model in ("gemini", "copilot", "codex") else "auditor"
        return self._emit(role, model, MessageType.CLAIM, content, file, scenario)

    def evidence(self, model: str, content: str, file: Optional[str] = None,
                 scenario: Optional[str] = None) -> ReasoningMessage:
        """
        Emit EVIDENCE: A reference to concrete artifacts supporting a claim.

        Example: "Changes in notus.rs, detection.rs."
        """
        role = "generator" if model in ("gemini", "copilot", "codex") else "auditor"
        return self._emit(role, model, MessageType.EVIDENCE, content, file, scenario)

    def challenge(self, model: str, content: str, file: Optional[str] = None,
                  scenario: Optional[str] = None) -> ReasoningMessage:
        """
        Emit CHALLENGE: Cannot validate a claim from available evidence.

        Example: "No benchmark data supports performance invariance."
        """
        role = "auditor"  # Challenges typically come from auditors
        return self._emit(role, model, MessageType.CHALLENGE, content, file, scenario)

    def decision(self, content: str, file: Optional[str] = None,
                 scenario: Optional[str] = None) -> ReasoningMessage:
        """
        Emit DECISION: System explains what it decided and why.

        Only the system may emit DECISION messages.

        Example: "All scenarios classified as @trusted. No hallucinations detected."
        """
        return self._emit("system", "system", MessageType.DECISION, content, file, scenario)

    # -------------------- Decision Boundary Emitters --------------------

    def emit_merge_decision(self, file_path: str, disagreement: float,
                            threshold: float, model_coverage: Dict[str, float],
                            scenario_count: int) -> None:
        """Emit reasoning at merge decision boundary."""
        coverage_str = ", ".join(f"{m}: {c*100:.0f}%" for m, c in model_coverage.items())

        if disagreement <= threshold:
            self.decision(
                f"Merged {scenario_count} scenarios with {disagreement*100:.1f}% disagreement "
                f"(threshold: {threshold*100:.0f}%). Model coverage: {coverage_str}.",
                file=file_path
            )
        else:
            self.decision(
                f"DISPUTED: Models disagree ({disagreement*100:.1f}%) above threshold ({threshold*100:.0f}%). "
                f"Model coverage: {coverage_str}. Output generated but needs review.",
                file=file_path
            )

    def emit_classification_decision(self, file_path: str, scenario_name: str,
                                     classification: str, confidence: float,
                                     evidence_symbols: List[str],
                                     generating_models: List[str]) -> None:
        """Emit reasoning at classification decision boundary."""
        model_list = ", ".join(generating_models)
        evidence_str = ", ".join(evidence_symbols) if evidence_symbols else "none"

        self.decision(
            f"Classified as '{classification}' with {confidence:.1%} confidence. "
            f"Generated by: {model_list}. Evidence symbols: {evidence_str}.",
            file=file_path,
            scenario=scenario_name
        )

    def emit_audit_decision(self, file_path: str, audit_model: str,
                            trusted_count: int, needs_review_count: int,
                            hallucination_count: int,
                            findings: List[Dict]) -> None:
        """Emit reasoning at audit gate decision boundary."""
        if hallucination_count > 0:
            finding_summaries = [f"'{f.get('scenario', 'unknown')}'" for f in findings[:3]]
            self.decision(
                f"FLAGGED: Audit by {audit_model} found {hallucination_count} potential hallucinations. "
                f"Scenarios: {', '.join(finding_summaries)}. Output needs review.",
                file=file_path
            )
        elif needs_review_count > 0:
            self.decision(
                f"Audit by {audit_model} completed. {trusted_count} trusted, "
                f"{needs_review_count} need review, {hallucination_count} hallucinations.",
                file=file_path
            )
        else:
            self.decision(
                f"Audit by {audit_model} passed. All {trusted_count} scenarios classified as trusted.",
                file=file_path
            )

        # Emit challenges for each finding
        for finding in findings:
            self.challenge(
                audit_model,
                f"Cannot validate claim: {finding.get('claim', 'unspecified')}",
                file=file_path,
                scenario=finding.get("scenario")
            )

    def emit_gate_decision(self, file_path: str, passed: bool,
                           reason: str, is_dispute: bool = False) -> None:
        """Emit reasoning at final gate decision boundary."""
        if passed:
            self.decision(f"Gate passed: {reason}", file=file_path)
        elif is_dispute:
            self.decision(f"Gate disputed: {reason}", file=file_path)
        else:
            self.decision(f"Gate failed: {reason}", file=file_path)

    # -------------------- Persistence --------------------

    def persist_to_db(self, db: 'TelemetryDB') -> int:
        """
        Persist all messages to the telemetry database.

        Note: Telemetry is memory, not policy. The DB never emits messages back.

        Returns count of messages persisted.
        """
        count = 0
        for msg in self.messages:
            try:
                db.conn.execute("""
                    INSERT INTO reasoning_messages
                    (message_id, run_id, timestamp, role, model, type, content, file, scenario)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    msg.run_id,
                    msg.timestamp,
                    msg.role,
                    msg.model,
                    msg.msg_type,
                    msg.content,
                    msg.file,
                    msg.scenario,
                ))
                count += 1
            except Exception as e:
                sys.stderr.write(f"[reasoning] Failed to persist message: {e}\n")
        db.conn.commit()
        return count

    def get_transcript(self) -> str:
        """Get human-readable transcript of all messages."""
        return "\n".join(msg.to_line() for msg in self.messages)

    def get_messages_by_type(self, msg_type: str) -> List[ReasoningMessage]:
        """Get all messages of a specific type."""
        return [m for m in self.messages if m.msg_type == msg_type]

    def get_decisions(self) -> List[ReasoningMessage]:
        """Get all DECISION messages."""
        return self.get_messages_by_type(MessageType.DECISION)

    def get_challenges(self) -> List[ReasoningMessage]:
        """Get all CHALLENGE messages."""
        return self.get_messages_by_type(MessageType.CHALLENGE)


# Global reasoning bus instance (set in main)
_reasoning_bus: Optional[ReasoningBus] = None


def get_reasoning_bus() -> Optional[ReasoningBus]:
    """Get the global reasoning bus instance."""
    return _reasoning_bus


# ----------------------------- Wizard -----------------------------

def prompt_choice(question: str, options: List[tuple], default: int = 0) -> str:
    """Prompt user for a choice from options. Returns the value of selected option."""
    print(f"\n{question}")
    for i, (label, value, hint) in enumerate(options):
        marker = "→" if i == default else " "
        print(f"  {marker} [{i+1}] {label}")
        if hint:
            print(f"        {hint}")

    while True:
        try:
            choice = input(f"\nChoice [1-{len(options)}] (default: {default+1}): ").strip()
            if not choice:
                return options[default][1]
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
            print(f"Please enter 1-{len(options)}")
        except ValueError:
            print(f"Please enter 1-{len(options)}")
        except (KeyboardInterrupt, EOFError):
            print("\nWizard cancelled.")
            raise SystemExit(1)


def prompt_yn(question: str, default: bool = True) -> bool:
    """Prompt for yes/no. Returns boolean."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print("\nWizard cancelled.")
        raise SystemExit(1)


def run_wizard(repo: str, pr: int) -> Dict[str, Any]:
    """
    Interactive wizard that generates CLI configuration.
    Returns a dict of settings to apply.
    """
    print("=" * 60)
    print("  Gherkin Consensus Generator - Configuration Wizard")
    print("=" * 60)
    print(f"\n  Repo: {repo}")
    print(f"  PR:   #{pr}")

    config = {
        "repo": repo,
        "pr": pr,
        "audit": False,
        "parallel": 1,
        "threshold": DEFAULT_DISAGREEMENT_THRESHOLD,
        "fail_fast": False,
        "timestamp": True,
        "sdlc_checks": False,
        "no_disagreement_gate": False,
        "generation_models": DEFAULT_GENERATION_MODELS.copy(),
        "audit_model": DEFAULT_AUDIT_MODEL,
    }

    # Step 1: PR Size/Type
    pr_type = prompt_choice(
        "Step 1/5: What kind of PR is this?",
        [
            ("Small bugfix (≤5 files)", "small", "Fast, light parallelism"),
            ("Medium change (6-15 files)", "medium", "Balanced settings"),
            ("Large refactor (16+ files)", "large", "High parallelism, relaxed thresholds"),
            ("Crypto / protocol / parser", "sensitive", "Extra audit scrutiny"),
        ],
        default=1  # medium
    )

    # Step 2: Optimization goal
    goal = prompt_choice(
        "Step 2/5: What are you optimizing for?",
        [
            ("Speed", "speed", "Minimal checks, max parallelism"),
            ("Safety (hallucination detection)", "safety", "Full audit, strict thresholds"),
            ("Coverage exploration", "coverage", "Relaxed disagreement, see all scenarios"),
            ("CI gating", "ci", "Fail-fast, strict, deterministic"),
        ],
        default=1  # safety
    )

    # Step 3: Environment
    env = prompt_choice(
        "Step 3/5: Where are you running this?",
        [
            ("Laptop (interactive)", "laptop", "Moderate parallelism, watch memory"),
            ("CI server", "ci_server", "High parallelism, no prompts"),
            ("Rate-limited / slow connection", "limited", "Low parallelism, longer timeouts"),
        ],
        default=0  # laptop
    )

    # Step 4: Failure tolerance
    failure_mode = prompt_choice(
        "Step 4/5: How should failures be handled?",
        [
            ("Continue and summarize all", "continue", "See everything, decide at end"),
            ("Stop on first hard failure", "stop", "Fail-fast for CI pipelines"),
        ],
        default=0  # continue
    )

    # Step 5: Model selection
    model_choice = prompt_choice(
        "Step 5/5: Which models should participate?",
        [
            ("Gemini + Copilot (default)", "default", "Standard dual-model consensus"),
            ("Gemini + Copilot + Codex", "with_codex", "Add structural coverage from Codex"),
            ("All models (Gemini + Copilot + Codex)", "all_gen", "Maximum coverage, slower"),
            ("Custom selection", "custom", "Choose specific models"),
        ],
        default=0
    )

    # Map choices to config

    # PR type settings
    if pr_type == "small":
        config["parallel"] = 2
        config["threshold"] = 0.35
    elif pr_type == "medium":
        config["parallel"] = 4
        config["threshold"] = 0.40
    elif pr_type == "large":
        config["parallel"] = 6
        config["threshold"] = 0.50
    elif pr_type == "sensitive":
        config["parallel"] = 2
        config["audit"] = True
        config["threshold"] = 0.30
        config["sdlc_checks"] = True

    # Goal overrides
    if goal == "speed":
        config["audit"] = False
        config["parallel"] = max(config["parallel"], 6)
        config["no_disagreement_gate"] = True
    elif goal == "safety":
        config["audit"] = True
        config["threshold"] = min(config["threshold"], 0.35)
    elif goal == "coverage":
        config["audit"] = True
        config["threshold"] = 0.80
        config["no_disagreement_gate"] = True
    elif goal == "ci":
        config["audit"] = True
        config["fail_fast"] = True
        config["threshold"] = 0.35

    # Environment overrides
    if env == "laptop":
        config["parallel"] = min(config["parallel"], 4)
    elif env == "ci_server":
        config["parallel"] = max(config["parallel"], 4)
    elif env == "limited":
        config["parallel"] = 1

    # Failure mode
    if failure_mode == "stop":
        config["fail_fast"] = True

    # Model selection
    if model_choice == "default":
        config["generation_models"] = ["gemini", "copilot"]
    elif model_choice == "with_codex":
        config["generation_models"] = ["gemini", "copilot", "codex"]
    elif model_choice == "all_gen":
        config["generation_models"] = ["gemini", "copilot", "codex"]
    elif model_choice == "custom":
        print("\n  Available generation models: gemini, copilot, codex")
        custom_models = input("  Enter models (comma-separated): ").strip()
        if custom_models:
            config["generation_models"] = [m.strip() for m in custom_models.split(",") if m.strip()]
        # Ask about audit model
        print(f"\n  Current audit model: {config['audit_model']}")
        custom_audit = input("  Change audit model? (enter model name or press Enter to keep): ").strip()
        if custom_audit:
            config["audit_model"] = custom_audit

    # If audit is enabled and using Claude for audit, make sure it's set
    if config["audit"] and config["audit_model"] == "claude":
        pass  # Already set correctly

    return config


def config_to_cli(config: Dict[str, Any]) -> List[str]:
    """Convert config dict to CLI arguments list."""
    cmd = [
        sys.argv[0],
        "--repo", config["repo"],
        "--pr", str(config["pr"]),
    ]

    if config.get("audit"):
        cmd.append("--audit")
    if config.get("parallel", 1) > 1:
        cmd.extend(["--parallel", str(config["parallel"])])
    if config.get("threshold") != DEFAULT_DISAGREEMENT_THRESHOLD:
        cmd.extend(["--threshold", str(config["threshold"])])
    if config.get("fail_fast"):
        cmd.append("--fail-fast")
    if config.get("timestamp"):
        cmd.append("--timestamp")
    if config.get("sdlc_checks"):
        cmd.append("--sdlc-checks")
    if config.get("no_disagreement_gate"):
        cmd.append("--no-disagreement-gate")

    # Model configuration
    gen_models = config.get("generation_models", DEFAULT_GENERATION_MODELS)
    if gen_models != DEFAULT_GENERATION_MODELS:
        cmd.extend(["--models", ",".join(gen_models)])

    audit_model = config.get("audit_model", DEFAULT_AUDIT_MODEL)
    if audit_model != DEFAULT_AUDIT_MODEL:
        cmd.extend(["--audit-model", audit_model])

    return cmd


def print_wizard_summary(config: Dict[str, Any]) -> None:
    """Print human-readable summary of wizard choices."""
    print("\n" + "=" * 60)
    print("  Configuration Summary")
    print("=" * 60)

    gen_models = config.get('generation_models', DEFAULT_GENERATION_MODELS)
    audit_model = config.get('audit_model', DEFAULT_AUDIT_MODEL)

    print(f"""
  Target:      {config['repo']} PR #{config['pr']}
  Parallelism: {config['parallel']} workers

  Models:
    Generation: {', '.join(gen_models)}
    Audit:      {audit_model if config['audit'] else '(disabled)'}

  Audit:       {'Enabled (cross-model hallucination check)' if config['audit'] else 'Disabled'}
  Threshold:   {config['threshold']*100:.0f}% disagreement allowed
  Fail-fast:   {'Yes (stop on first failure)' if config['fail_fast'] else 'No (continue and summarize)'}
  SDLC checks: {'Enabled' if config['sdlc_checks'] else 'Disabled'}
""")

    # Derived policy explanation
    print("  Failure Policy:")
    if config.get("no_disagreement_gate"):
        print("    • Disagreement is advisory only (won't fail the run)")
    else:
        print(f"    • Disagreement >{config['threshold']*100:.0f}% marks file as failed")
    if config["audit"]:
        print("    • High-confidence hallucinations fail the file")
    print("    • Transport errors (timeout, empty output) always fail")

    # Estimate runtime
    est_per_file = 60 if config["audit"] else 30  # seconds
    est_total = est_per_file * 10 / max(1, config["parallel"])  # assume ~10 files
    print(f"\n  Estimated runtime: ~{int(est_total/60)}-{int(est_total/60)+5} minutes for a typical PR")


# ----------------------------- Utilities -----------------------------

def die(msg: str, code: int) -> None:
    sys.stderr.write(f"[error] {msg}\n")
    raise SystemExit(code)


def run_cmd(cmd: List[str], *, input_text: Optional[str] = None, timeout: int = 180) -> str:
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


# ----------------------------- Model Invocation with Observability -----------------------------

class FailureType:
    """Explicit failure classification."""
    SUCCESS = "SUCCESS"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    TIMEOUT = "TIMEOUT"
    NONZERO_EXIT = "NONZERO_EXIT"
    MALFORMED_GHERKIN = "MALFORMED_GHERKIN"
    UNKNOWN = "UNKNOWN"


class OutcomeStatus:
    """
    Outcome classification for file processing.
    Distinguishes between actual failures and consensus issues.
    """
    SUCCESS = "success"           # Generated and passed all gates
    DISPUTED = "disputed"         # Generated but models disagreed (above threshold)
    FAILED = "failed"             # Actual error (parse, timeout, API, etc.)
    FLAGGED = "flagged"           # Audit flagged too many hallucinations


@dataclass
class ModelResult:
    """Detailed result from a model invocation."""
    model: str
    stdout: str
    stderr: str
    returncode: int
    timeout: bool
    failure_type: str
    retry_count: int = 0


def run_model_cmd(cmd: List[str], model: str, timeout: int = 180, max_retries: int = 1) -> ModelResult:
    """
    Run a model CLI command with full observability.
    Retries once on empty output only.
    """
    for retry in range(max_retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=timeout,
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            rc = proc.returncode

            # Classify the result
            if rc != 0:
                return ModelResult(
                    model=model, stdout=stdout, stderr=stderr,
                    returncode=rc, timeout=False,
                    failure_type=FailureType.NONZERO_EXIT, retry_count=retry
                )

            if not stdout.strip():
                if retry < max_retries:
                    sys.stderr.write(f"[retry] {model} returned empty output, retrying...\n")
                    continue
                return ModelResult(
                    model=model, stdout=stdout, stderr=stderr,
                    returncode=rc, timeout=False,
                    failure_type=FailureType.EMPTY_OUTPUT, retry_count=retry
                )

            return ModelResult(
                model=model, stdout=stdout, stderr=stderr,
                returncode=rc, timeout=False,
                failure_type=FailureType.SUCCESS, retry_count=retry
            )

        except subprocess.TimeoutExpired:
            return ModelResult(
                model=model, stdout="", stderr=f"Command timed out after {timeout}s",
                returncode=-1, timeout=True,
                failure_type=FailureType.TIMEOUT, retry_count=retry
            )
        except Exception as e:
            return ModelResult(
                model=model, stdout="", stderr=str(e),
                returncode=-1, timeout=False,
                failure_type=FailureType.UNKNOWN, retry_count=retry
            )

    # Should not reach here
    return ModelResult(
        model=model, stdout="", stderr="Max retries exceeded",
        returncode=-1, timeout=False,
        failure_type=FailureType.UNKNOWN, retry_count=max_retries
    )


def run_model(model: str, prompt: str, max_retries: int = 1) -> ModelResult:
    """
    Unified model runner using the registry.
    Builds the correct CLI command based on model spec.
    """
    spec = get_model_spec(model)

    if spec.prompt_via_stdin:
        # Models like Codex that take prompt via stdin
        try:
            for retry in range(max_retries + 1):
                proc = subprocess.run(
                    spec.cli_cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=spec.timeout,
                )
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                rc = proc.returncode

                if rc != 0:
                    return ModelResult(
                        model=model, stdout=stdout, stderr=stderr,
                        returncode=rc, timeout=False,
                        failure_type=FailureType.NONZERO_EXIT, retry_count=retry
                    )
                if not stdout.strip():
                    if retry < max_retries:
                        sys.stderr.write(f"[retry] {model} returned empty output, retrying...\n")
                        continue
                    return ModelResult(
                        model=model, stdout=stdout, stderr=stderr,
                        returncode=rc, timeout=False,
                        failure_type=FailureType.EMPTY_OUTPUT, retry_count=retry
                    )
                return ModelResult(
                    model=model, stdout=stdout, stderr=stderr,
                    returncode=rc, timeout=False,
                    failure_type=FailureType.SUCCESS, retry_count=retry
                )
        except subprocess.TimeoutExpired:
            return ModelResult(
                model=model, stdout="", stderr=f"Command timed out after {spec.timeout}s",
                returncode=-1, timeout=True,
                failure_type=FailureType.TIMEOUT, retry_count=max_retries
            )
        except Exception as e:
            return ModelResult(
                model=model, stdout="", stderr=str(e),
                returncode=-1, timeout=False,
                failure_type=FailureType.UNKNOWN, retry_count=max_retries
            )
    else:
        # Models like Gemini/Copilot/Claude that take prompt as argument
        cmd = spec.cli_cmd + [prompt]
        return run_model_cmd(cmd, model, timeout=spec.timeout, max_retries=max_retries)

    # Should not reach here
    return ModelResult(
        model=model, stdout="", stderr="Unknown error",
        returncode=-1, timeout=False,
        failure_type=FailureType.UNKNOWN, retry_count=max_retries
    )


def write_error_log(out_dir: str, base: str, model: str, file_path: str,
                    result: ModelResult, reason: str) -> None:
    """Write a sidecar error log for failed model invocations."""
    error_data = {
        "model": model,
        "file": file_path,
        "reason": reason,
        "failure_type": result.failure_type,
        "returncode": result.returncode,
        "stderr": result.stderr[:1000] if result.stderr else "",
        "stdout_len": len(result.stdout) if result.stdout else 0,
        "timeout": result.timeout,
        "retry_count": result.retry_count,
    }
    error_path = os.path.join(out_dir, f"{base}.{model}.error.json")
    with open(error_path, "w") as f:
        json.dump(error_data, f, indent=2)
    sys.stderr.write(f"[MODEL_ERROR] {model} {file_path}: {reason}\n")
    if result.stderr:
        sys.stderr.write(f"    stderr: {result.stderr[:200]}\n")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^\w]+", "_", s).strip("_") or "artifact"

# ----------------------------- GH Tool -----------------------------

@tool
def gh_pr_bundle(repo: str, pr: int) -> dict:
    """
    Fetch a GitHub pull request patch and contextual metadata using the GitHub CLI.

    Inputs:
      repo: Repository in 'org/repo' format.
      pr: Pull request number.

    Returns:
      A dictionary with keys:
        - repo: repository name
        - pr: pull request number
        - patch: unified diff text
        - context: PR metadata including title, body, comments, reviews, author, and URL
    """
    patch = run_cmd(["gh", "pr", "diff", str(pr), "--repo", repo])
    ctx = json.loads(run_cmd([
        "gh", "pr", "view", str(pr),
        "--repo", repo,
        "--json", "title,body,comments,reviews,author,url"
    ]))
    return {"repo": repo, "pr": pr, "patch": patch, "context": ctx}

# ----------------------------- Patch Analysis -----------------------------

class PatchAnalyzer:
    def __init__(self, patch: str):
        self.ps = PatchSet(patch)

    def summarize(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for f in self.ps:
            changes = []
            for h in f:
                a = sum(1 for l in h if l.is_added)
                r = sum(1 for l in h if l.is_removed)
                if a or r:
                    changes.append(f"{a} added / {r} removed")
            if changes:
                out[f.path] = changes
        return out

    def extract_evidence(self) -> Dict[str, Any]:
        """Extract evidence index for hallucination checking."""
        evidence = {
            "files_changed": [],
            "symbols_added": [],
            "symbols_removed": [],
            "interfaces_touched": [],
            "raw_additions": [],
            "raw_removals": [],
        }

        for f in self.ps:
            evidence["files_changed"].append(f.path)

            for hunk in f:
                for line in hunk:
                    text = str(line.value).strip()
                    if not text or text.startswith("//") or text.startswith("#"):
                        continue

                    if line.is_added:
                        evidence["raw_additions"].append(text)
                        # Extract symbols (fn, struct, trait, impl, pub, def, class)
                        for pattern in [r'\bfn\s+(\w+)', r'\bstruct\s+(\w+)', r'\btrait\s+(\w+)',
                                       r'\bimpl\s+(\w+)', r'\bpub\s+(?:fn|struct|trait)\s+(\w+)',
                                       r'\bdef\s+(\w+)', r'\bclass\s+(\w+)']:
                            matches = re.findall(pattern, text)
                            evidence["symbols_added"].extend(matches)

                    elif line.is_removed:
                        evidence["raw_removals"].append(text)
                        for pattern in [r'\bfn\s+(\w+)', r'\bstruct\s+(\w+)', r'\btrait\s+(\w+)',
                                       r'\bimpl\s+(\w+)', r'\bpub\s+(?:fn|struct|trait)\s+(\w+)',
                                       r'\bdef\s+(\w+)', r'\bclass\s+(\w+)']:
                            matches = re.findall(pattern, text)
                            evidence["symbols_removed"].extend(matches)

        # Deduplicate
        for key in ["symbols_added", "symbols_removed"]:
            evidence[key] = list(set(evidence[key]))

        return evidence


def format_evidence(evidence: Dict[str, Any], context: dict) -> str:
    """Format evidence index for audit prompts."""
    lines = ["## Evidence Index", ""]
    lines.append(f"**Files changed:** {', '.join(evidence['files_changed'])}")

    if evidence["symbols_added"]:
        lines.append(f"**Symbols added:** {', '.join(evidence['symbols_added'])}")
    if evidence["symbols_removed"]:
        lines.append(f"**Symbols removed:** {', '.join(evidence['symbols_removed'])}")

    # Include PR context
    if context.get("title"):
        lines.append(f"**PR title:** {context['title']}")
    if context.get("body"):
        lines.append(f"**PR description:** {context['body'][:500]}")

    # Sample of actual code changes
    if evidence["raw_additions"][:5]:
        lines.append("**Sample additions:**")
        for line in evidence["raw_additions"][:5]:
            lines.append(f"  + {line[:100]}")
    if evidence["raw_removals"][:5]:
        lines.append("**Sample removals:**")
        for line in evidence["raw_removals"][:5]:
            lines.append(f"  - {line[:100]}")

    return "\n".join(lines)

# ----------------------------- Hint Distillation -----------------------------

@dataclass
class Hint:
    text: str
    weight: float

def distill_hints(ctx: dict) -> List[Hint]:
    hints: List[Hint] = []
    body = (ctx.get("body") or "").strip()
    if body:
        hints.append(Hint(body, 1.0))

    for c in ctx.get("comments", []) + ctx.get("reviews", []):
        text = (c.get("body") or "").strip()
        if any(t in text.lower() for t in HIGH_SIGNAL_TERMS):
            hints.append(Hint(text, 0.8))

    hints.sort(key=lambda h: h.weight, reverse=True)
    return hints[:10]

def format_hints(hints: List[Hint]) -> str:
    return "\n".join(f"- {h.text[:400]}" for h in hints) if hints else "- (none)"

# ----------------------------- Gherkin Model -----------------------------

@dataclass
class Scenario:
    name: str
    steps: List[str]
    tags: List[str]

@dataclass
class Feature:
    name: str
    scenarios: Dict[str, Scenario]
    background: List[str] = None

    def __post_init__(self):
        if self.background is None:
            self.background = []

class GherkinParser:
    STEP_PREFIXES = ("Given", "When", "Then", "And", "But")

    @staticmethod
    def parse(text: str) -> Feature:
        fname = None
        scenarios: Dict[str, Scenario] = {}
        background_steps: List[str] = []
        current: Optional[Scenario] = None
        in_background = False

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("Feature:"):
                fname = line[8:].strip()
                continue

            if line.startswith("Background:"):
                in_background = True
                current = None
                continue

            if line.startswith("Scenario:"):
                in_background = False
                name = line[9:].strip()
                current = Scenario(name, [], [])
                scenarios[name] = current
                continue

            if line.startswith(GherkinParser.STEP_PREFIXES):
                if in_background:
                    background_steps.append(line)
                elif current:
                    current.steps.append(line)

        if not fname or not scenarios:
            raise ValueError("Invalid Gherkin")
        return Feature(fname, scenarios, background_steps)


def apply_background(feature: Feature) -> Feature:
    """Normalize Background steps into each Scenario for fair comparison."""
    if not feature.background:
        return feature

    for sc in feature.scenarios.values():
        sc.steps = feature.background + sc.steps
    return feature


def is_background_step(step: str) -> bool:
    """Identify setup/invariant steps that shouldn't count as disagreement."""
    lower = step.lower()
    if not lower.startswith("given"):
        return False
    invariant_keywords = (
        "initialized", "configured", "legacy", "module",
        "exists", "loaded", "setup", "environment", "state"
    )
    return any(kw in lower for kw in invariant_keywords)

# ----------------------------- LLM Tools -----------------------------

def clean_llm_output(text: str) -> str:
    """Clean LLM output to extract just the Gherkin content."""
    lines = text.splitlines()
    cleaned = []
    found_feature = False

    for line in lines:
        stripped = line.strip()

        # Skip common noise lines
        if stripped.startswith("Loaded cached credentials"):
            continue
        if stripped.startswith(("Total usage", "Total duration", "Total code")):
            continue
        if stripped.startswith("Usage by model"):
            break  # Stop at usage stats
        # Skip Copilot tool execution logs
        if stripped.startswith(("✓", "✗", "$", "└")):
            continue
        if "lines read" in stripped or "files found" in stripped:
            continue

        # Handle markdown code blocks
        if stripped.startswith("```"):
            continue

        # Start capturing at Feature:
        if stripped.startswith("Feature:"):
            found_feature = True

        if found_feature:
            cleaned.append(line)

    return "\n".join(cleaned).strip()

@tool
def gemini_gherkin(prompt: str) -> str:
    """Generate Gherkin scenarios using Gemini AI from the given prompt."""
    output = run_cmd(["gemini", "-p", prompt])
    return clean_llm_output(output)

@tool
def copilot_gherkin(prompt: str) -> str:
    """Generate Gherkin scenarios using GitHub Copilot from the given prompt."""
    # Don't allow tools - we just want text output
    output = run_cmd(["copilot", "-p", prompt])
    return clean_llm_output(output)

# ----------------------------- Cross-Audit (Hallucination Detection) -----------------------------

AUDIT_PROMPT_TEMPLATE = """You are auditing Gherkin regression tests for unsupported claims.

Task:
- Identify any Scenario or step that asserts behavior, guarantees, or constraints
  that are NOT supported by the provided code changes or project context.
- Do NOT evaluate test quality.
- Do NOT comment on completeness.
- Do NOT suggest additional scenarios.
- Only judge whether claims are supported by evidence.

{evidence}

## Tests to audit:
{gherkin}

## Output format:
If you find unsupported claims, list them as:
UNSUPPORTED: [Scenario name]: [specific unsupported claim]

If all claims are supported by evidence, output exactly:
NONE

Output ONLY the list or NONE. No explanations."""


class ClaimClass:
    """Classification of claim groundedness."""
    GROUNDED = "grounded"      # Direct symbol/behavior in diff
    INFERRED = "inferred"      # Logical consequence of grounded change
    CONTEXTUAL = "contextual"  # Domain-valid but outside diff
    SPECULATIVE = "speculative"  # Plausible but unsupported
    HALLUCINATED = "hallucinated"  # Invented / false


@dataclass
class ClassifiedClaim:
    """A scenario claim with classification and confidence metadata."""
    scenario: str
    classification: str  # ClaimClass value
    evidence_symbols: List[str]  # Symbols from diff that support this
    generating_models: List[str]  # Models that produced this scenario
    confidence: float  # 0.0 - 1.0 based on evidence + agreement
    layer: str  # "api", "implementation", "user-observable", "unknown"

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "classification": self.classification,
            "evidence_symbols": self.evidence_symbols,
            "generating_models": self.generating_models,
            "confidence": self.confidence,
            "layer": self.layer,
            "agreement_count": len(self.generating_models),
        }


@dataclass
class AuditFinding:
    scenario: str
    claim: str
    auditor: str  # "gemini", "copilot", "claude", etc.


def detect_layer(scenario_text: str) -> str:
    """Detect which layer a scenario tests: api, implementation, or user-observable."""
    text = scenario_text.lower()

    # API layer indicators
    if any(w in text for w in ["response", "status", "endpoint", "request", "api", "http", "json", "return"]):
        return "api"

    # Implementation layer indicators
    if any(w in text for w in ["process", "running", "started", "initialized", "configured", "loaded", "memory"]):
        return "implementation"

    # User-observable layer indicators
    if any(w in text for w in ["results", "output", "display", "available", "accessible", "visible", "user"]):
        return "user-observable"

    return "unknown"


def classify_claim(scenario_name: str, scenario_steps: List[str],
                   evidence: Dict[str, Any], generating_models: List[str]) -> ClassifiedClaim:
    """
    Classify a scenario claim based on evidence and agreement.
    Returns ClassifiedClaim with confidence score.
    """
    # Extract symbols from evidence
    added_symbols = set(evidence.get("added", []))
    removed_symbols = set(evidence.get("removed", []))
    all_symbols = added_symbols | removed_symbols

    # Check which symbols appear in the scenario
    scenario_text = scenario_name.lower() + " " + " ".join(s.lower() for s in scenario_steps)
    matched_symbols = [s for s in all_symbols if s.lower() in scenario_text]

    # Determine classification based on evidence match
    if matched_symbols:
        if len(matched_symbols) >= 2:
            classification = ClaimClass.GROUNDED
        else:
            classification = ClaimClass.INFERRED
    else:
        # Check for domain keywords without direct evidence
        domain_keywords = ["test", "build", "run", "execute", "make", "docker", "container"]
        if any(kw in scenario_text for kw in domain_keywords):
            classification = ClaimClass.CONTEXTUAL
        else:
            classification = ClaimClass.SPECULATIVE

    # Detect layer
    layer = detect_layer(scenario_text)

    # Calculate confidence: evidence weight + agreement multiplier
    evidence_score = min(1.0, len(matched_symbols) * 0.3)  # 0.3 per symbol, max 1.0
    agreement_multiplier = 1.0 + (len(generating_models) - 1) * 0.2  # +0.2 per additional model
    base_confidence = {
        ClaimClass.GROUNDED: 0.9,
        ClaimClass.INFERRED: 0.7,
        ClaimClass.CONTEXTUAL: 0.5,
        ClaimClass.SPECULATIVE: 0.3,
        ClaimClass.HALLUCINATED: 0.1,
    }.get(classification, 0.5)

    confidence = min(1.0, (base_confidence + evidence_score) * agreement_multiplier / 2)

    return ClassifiedClaim(
        scenario=scenario_name,
        classification=classification,
        evidence_symbols=matched_symbols,
        generating_models=generating_models,
        confidence=round(confidence, 3),
        layer=layer,
    )


def parse_audit_response(response: str, auditor: str) -> List[AuditFinding]:
    """Parse audit response into structured findings."""
    findings = []
    response = response.strip()

    if response.upper() == "NONE" or not response:
        return findings

    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("UNSUPPORTED:"):
            content = line[len("UNSUPPORTED:"):].strip()
            if ":" in content:
                scenario, claim = content.split(":", 1)
                findings.append(AuditFinding(scenario.strip(), claim.strip(), auditor))
            else:
                findings.append(AuditFinding("Unknown", content, auditor))

    return findings


def cross_audit(gemini_gherkin_text: str, copilot_gherkin_text: str,
                evidence_text: str) -> Dict[str, List[AuditFinding]]:
    """Have each model audit the other's output for unsupported claims."""
    results = {"gemini_findings": [], "copilot_findings": []}

    # Gemini audits Copilot's tests
    gemini_audit_prompt = AUDIT_PROMPT_TEMPLATE.format(
        evidence=evidence_text,
        gherkin=copilot_gherkin_text
    )
    try:
        gemini_response = run_cmd(["gemini", "-p", gemini_audit_prompt])
        gemini_response = clean_llm_output(gemini_response)
        results["gemini_findings"] = parse_audit_response(gemini_response, "gemini")
    except Exception as e:
        sys.stderr.write(f"[warn] Gemini audit failed: {e}\n")

    # Copilot audits Gemini's tests
    copilot_audit_prompt = AUDIT_PROMPT_TEMPLATE.format(
        evidence=evidence_text,
        gherkin=gemini_gherkin_text
    )
    try:
        copilot_response = run_cmd(["copilot", "-p", copilot_audit_prompt])
        copilot_response = clean_llm_output(copilot_response)
        results["copilot_findings"] = parse_audit_response(copilot_response, "copilot")
    except Exception as e:
        sys.stderr.write(f"[warn] Copilot audit failed: {e}\n")

    return results


def apply_hallucination_gate(feature: Feature, audit_results: Dict[str, List[AuditFinding]]) -> Dict[str, Any]:
    """Apply hallucination gate: tag scenarios based on audit findings."""
    # Collect all findings from all auditors (handles both cross-audit and dedicated audit)
    all_findings = []
    for key, findings in audit_results.items():
        if key.endswith("_findings"):
            all_findings.extend(findings)

    # Count flags per scenario
    scenario_flags: Dict[str, List[str]] = {}
    for finding in all_findings:
        scenario_name = finding.scenario
        if scenario_name not in scenario_flags:
            scenario_flags[scenario_name] = []
        scenario_flags[scenario_name].append(finding.auditor)

    # Determine threshold for high-confidence (2+ for cross-audit, 1 for dedicated audit)
    num_auditors = len([k for k in audit_results.keys() if k.endswith("_findings")])
    high_confidence_threshold = 2 if num_auditors >= 2 else 1

    # Categorize scenarios
    high_confidence_hallucinations = []  # Multiple auditors flagged (or single dedicated auditor)
    needs_review = []  # One auditor flagged (in cross-audit mode)
    trusted = []  # No auditor flagged

    for name, sc in feature.scenarios.items():
        flags = scenario_flags.get(name, [])
        unique_flaggers = set(flags)

        if len(unique_flaggers) >= high_confidence_threshold:
            high_confidence_hallucinations.append(name)
            sc.tags.append("@hallucination")
        elif len(unique_flaggers) >= 1:
            needs_review.append(name)
            sc.tags.append("@needs-review")
        else:
            trusted.append(name)
            sc.tags.append("@trusted")

    return {
        "high_confidence_hallucinations": high_confidence_hallucinations,
        "needs_review": needs_review,
        "trusted": trusted,
        "all_findings": all_findings,
    }

# ----------------------------- Consensus -----------------------------

def detect_domain(path: str) -> str:
    p = path.lower()
    if "/api/" in p or "controller" in p:
        return "api"
    if "docker" in p or "ci" in p:
        return "infra"
    return "default"

def merge_features(g: Feature, c: Feature, domain: str) -> dict:
    """Legacy 2-model merge for backward compatibility."""
    return merge_multiple_features([("gemini", g), ("copilot", c)], domain)


def merge_multiple_features(model_features: List[tuple], domain: str) -> dict:
    """
    Merge N features from different models.
    model_features: List of (model_name, Feature) tuples
    Returns: {"feature": Feature, "disagreement": float, "model_coverage": dict}
    """
    if not model_features:
        raise ValueError("No features to merge")

    if len(model_features) == 1:
        name, feature = model_features[0]
        return {
            "feature": feature,
            "disagreement": 0.0,
            "model_coverage": {name: 1.0},
        }

    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["default"])
    n_models = len(model_features)

    # Collect all scenario names and track which models have each
    scenario_models: Dict[str, List[str]] = {}
    scenario_steps: Dict[str, List[str]] = {}

    # Merge background steps from all models
    all_background = []
    for model_name, feature in model_features:
        all_background.extend(feature.background)
    merged_background = list(dict.fromkeys(all_background))

    # Collect scenarios from all models
    for model_name, feature in model_features:
        for name, scenario in feature.scenarios.items():
            if name not in scenario_models:
                scenario_models[name] = []
                scenario_steps[name] = []
            scenario_models[name].append(model_name)
            scenario_steps[name].extend(scenario.steps)

    # Build merged scenarios and calculate disagreement
    scenarios = {}
    conflicts = 0
    total = 0

    for name in scenario_models:
        # Deduplicate steps while preserving order
        steps = list(dict.fromkeys(scenario_steps[name]))
        behavioral_steps = [s for s in steps if not is_background_step(s)]

        # Count as conflict if not all models agree on this scenario
        models_with_scenario = len(scenario_models[name])
        if models_with_scenario < n_models:
            # Partial agreement - weight by how many models disagree
            conflict_weight = (n_models - models_with_scenario) / n_models
            conflicts += len(behavioral_steps) * conflict_weight

        total += len(behavioral_steps)
        scenarios[name] = Scenario(name, steps, [])

    disagreement = conflicts / max(1, total)

    # Pick feature name from highest-weighted model
    best_model = None
    best_weight = -1
    for model_name, feature in model_features:
        w = weights.get(model_name, 0.5)
        if w > best_weight:
            best_weight = w
            best_model = model_name
            preferred_name = feature.name

    # Calculate per-model coverage (what fraction of scenarios each model contributed)
    model_coverage = {}
    total_scenarios = len(scenario_models)
    for model_name, _ in model_features:
        count = sum(1 for models in scenario_models.values() if model_name in models)
        model_coverage[model_name] = count / max(1, total_scenarios)

    return {
        "feature": Feature(preferred_name, scenarios, merged_background),
        "disagreement": disagreement,
        "model_coverage": model_coverage,
        "scenario_models": scenario_models,  # Track which models produced each scenario
        "scenario_steps": {name: list(dict.fromkeys(steps)) for name, steps in scenario_steps.items()},
    }

# ----------------------------- SDLC Checks -----------------------------

@dataclass
class SDLCFinding:
    rule: str
    severity: str
    message: str

def sdlc_checks(feature: Feature, domain: str) -> List[SDLCFinding]:
    findings: List[SDLCFinding] = []
    steps = " ".join(
        s.lower() for sc in feature.scenarios.values() for s in sc.steps
    )

    if domain == "api" and "auth" in steps and "unauthorized" not in steps:
        findings.append(SDLCFinding(
            "SEC-AUTH", "fail", "Auth changes lack unauthorized test."
        ))

    if not findings:
        findings.append(SDLCFinding("SDLC-OK", "info", "No SDLC issues detected."))

    return findings

# ----------------------------- Rendering -----------------------------

def render_feature(feature: Feature, pr: int, file_path: str) -> str:
    out = [f"Feature: {feature.name}", f"# PR: {pr}", f"# File: {file_path}", ""]

    # Render Background section if present
    if feature.background:
        out.append("Background:")
        for step in feature.background:
            out.append(f"  {step}")
        out.append("")

    for sc in feature.scenarios.values():
        # Render tags if present
        if sc.tags:
            out.append(" ".join(sc.tags))
        out.append(f"Scenario: {sc.name}")
        for s in sc.steps:
            out.append(f"  {s}")
        out.append("")
    return "\n".join(out)

# ----------------------------- Worker Function -----------------------------

def process_single_file(item: dict, out_dir: str, pr: int, threshold: float,
                        do_sdlc: bool, do_audit: bool,
                        no_disagreement_gate: bool = False,
                        generation_models: List[str] = None,
                        audit_model: str = None) -> Dict[str, Any]:
    """
    Process a single file: generate Gherkin from N models, merge, audit, save.
    This function is designed to run in a worker process.
    """
    # Default to legacy behavior
    if generation_models is None:
        generation_models = DEFAULT_GENERATION_MODELS
    if audit_model is None:
        audit_model = DEFAULT_AUDIT_MODEL

    result = {
        "file": item["file"],
        "success": False,
        "failed": False,
        "error": None,
        "failure_type": None,
        "disagreement": 0.0,
        "hallucinations": [],
        "models_used": generation_models,
    }

    base = safe_name(item["file"])
    file_path = item["file"]

    # --- Step 1: Generate from all models ---
    model_outputs: Dict[str, str] = {}  # model -> cleaned output
    model_results: Dict[str, ModelResult] = {}  # model -> raw result

    for model_name in generation_models:
        model_result = run_model(model_name, item["prompt"])
        model_results[model_name] = model_result

        # Always write raw output first (even if empty/error)
        with open(os.path.join(out_dir, f"{base}.{model_name}.raw"), "w") as f:
            f.write(model_result.stdout)

        if model_result.failure_type != FailureType.SUCCESS:
            write_error_log(out_dir, base, model_name, file_path, model_result,
                            model_result.failure_type)
            result["error"] = f"{model_name}: {model_result.failure_type}"
            result["failure_type"] = model_result.failure_type
            result["failed"] = True
            return result

        model_outputs[model_name] = clean_llm_output(model_result.stdout)

    # --- Step 2: Parse Gherkin from all models ---
    model_features: List[tuple] = []  # List of (model_name, Feature)

    for model_name in generation_models:
        cleaned_output = model_outputs[model_name]
        try:
            feature = apply_background(GherkinParser.parse(cleaned_output))
            model_features.append((model_name, feature))
        except ValueError as e:
            # Write cleaned output for debugging
            with open(os.path.join(out_dir, f"{base}.{model_name}.cleaned"), "w") as f:
                f.write(cleaned_output)
            write_error_log(out_dir, base, model_name, file_path,
                            ModelResult(model_name, cleaned_output, str(e), 0, False,
                                       FailureType.MALFORMED_GHERKIN),
                            FailureType.MALFORMED_GHERKIN)
            result["error"] = f"{model_name}: {FailureType.MALFORMED_GHERKIN} - {e}"
            result["failure_type"] = FailureType.MALFORMED_GHERKIN
            result["failed"] = True
            return result

    # --- Step 3: Merge and evaluate ---
    try:
        domain = detect_domain(file_path)
        merged = merge_multiple_features(model_features, domain)
        result["disagreement"] = merged["disagreement"]
        result["model_coverage"] = merged.get("model_coverage", {})

        if merged["disagreement"] > threshold and not no_disagreement_gate:
            result["failed"] = True

        sdlc = []
        if do_sdlc:
            sdlc = sdlc_checks(merged["feature"], domain)
            if any(f.severity == "fail" for f in sdlc):
                result["failed"] = True

        # --- Step 4: Cross-audit hallucination detection ---
        if do_audit:
            evidence_text = format_evidence(item["evidence"], item["context"])

            # Use specified audit model if available, otherwise fall back to cross-audit
            if audit_model and audit_model not in generation_models:
                # Use dedicated audit model (e.g., Claude)
                audit_results = run_dedicated_audit(
                    merged["feature"], model_outputs, evidence_text, audit_model, out_dir, base
                )
            else:
                # Fall back to cross-audit between generation models
                # (for backward compatibility when audit_model is in generation_models)
                if len(generation_models) >= 2:
                    audit_results = cross_audit(
                        model_outputs[generation_models[0]],
                        model_outputs[generation_models[1]],
                        evidence_text
                    )
                else:
                    # Single model - no cross-audit possible
                    audit_results = []

            gate_result = apply_hallucination_gate(merged["feature"], audit_results)

            # --- Step 5: Classify claims and calculate confidence ---
            scenario_models = merged.get("scenario_models", {})
            scenario_steps = merged.get("scenario_steps", {})
            evidence = item.get("evidence", {})

            classified_claims = []
            for scenario_name, scenario in merged["feature"].scenarios.items():
                models = scenario_models.get(scenario_name, generation_models)
                steps = scenario_steps.get(scenario_name, scenario.steps)
                classified = classify_claim(scenario_name, steps, evidence, models)

                # Add classification-based tag (in addition to audit tag)
                if classified.classification == ClaimClass.GROUNDED:
                    scenario.tags.append(f"@grounded")
                elif classified.classification == ClaimClass.INFERRED:
                    scenario.tags.append(f"@inferred")
                elif classified.classification == ClaimClass.CONTEXTUAL:
                    scenario.tags.append(f"@contextual")
                elif classified.classification == ClaimClass.SPECULATIVE:
                    scenario.tags.append(f"@speculative")

                # Add layer tag
                if classified.layer != "unknown":
                    scenario.tags.append(f"@{classified.layer.replace('-', '_')}")

                classified_claims.append(classified)

            # Calculate model performance metrics for historical tracking
            model_performance = {}
            for model in generation_models:
                model_scenarios = [c for c in classified_claims if model in c.generating_models]
                if model_scenarios:
                    avg_confidence = sum(c.confidence for c in model_scenarios) / len(model_scenarios)
                    grounded_count = sum(1 for c in model_scenarios if c.classification == ClaimClass.GROUNDED)
                    model_performance[model] = {
                        "role": "generation",
                        "scenarios_produced": len(model_scenarios),
                        "avg_confidence": round(avg_confidence, 3),
                        "grounded_claims": grounded_count,
                        "grounded_ratio": round(grounded_count / len(model_scenarios), 3) if model_scenarios else 0,
                    }

            # Track audit model performance
            audit_model_used = audit_model if audit_model not in generation_models else "cross-audit"
            if audit_model_used != "cross-audit":
                model_performance[audit_model_used] = {
                    "role": "audit",
                    "findings_count": len(gate_result["all_findings"]),
                    "hallucinations_flagged": len(gate_result["high_confidence_hallucinations"]),
                    "scenarios_reviewed": len(merged["feature"].scenarios),
                }

            # Save audit results with confidence data
            with open(os.path.join(out_dir, f"{base}.audit.json"), "w") as f:
                json.dump({
                    "high_confidence_hallucinations": gate_result["high_confidence_hallucinations"],
                    "needs_review": gate_result["needs_review"],
                    "trusted": gate_result["trusted"],
                    "findings": [
                        {"scenario": f.scenario, "claim": f.claim, "auditor": f.auditor}
                        for f in gate_result["all_findings"]
                    ],
                    "audit_model": audit_model_used,
                    "classified_claims": [c.to_dict() for c in classified_claims],
                    "model_performance": model_performance,
                    "layer_coverage": {
                        "api": sum(1 for c in classified_claims if c.layer == "api"),
                        "implementation": sum(1 for c in classified_claims if c.layer == "implementation"),
                        "user_observable": sum(1 for c in classified_claims if c.layer == "user-observable"),
                        "unknown": sum(1 for c in classified_claims if c.layer == "unknown"),
                    },
                    "confidence_summary": {
                        "mean": round(sum(c.confidence for c in classified_claims) / max(1, len(classified_claims)), 3),
                        "min": round(min((c.confidence for c in classified_claims), default=0), 3),
                        "max": round(max((c.confidence for c in classified_claims), default=0), 3),
                    },
                }, f, indent=2)

            result["hallucinations"] = gate_result["high_confidence_hallucinations"]
            result["classified_claims"] = [c.to_dict() for c in classified_claims]
            result["model_performance"] = model_performance
            if gate_result["high_confidence_hallucinations"]:
                result["failed"] = True

        # Save merged feature
        with open(os.path.join(out_dir, f"{base}.feature"), "w") as f:
            f.write(render_feature(merged["feature"], pr, file_path))

        if sdlc:
            with open(os.path.join(out_dir, f"{base}_sdlc.json"), "w") as f:
                json.dump([f.__dict__ for f in sdlc], f, indent=2)

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["failed"] = True

    return result


def run_dedicated_audit(feature: Feature, model_outputs: Dict[str, str],
                        evidence_text: str, audit_model: str,
                        out_dir: str, base: str) -> Dict[str, List]:
    """
    Run a dedicated audit model (e.g., Claude) to audit the merged feature.
    Returns dict with audit findings in same format as cross_audit.
    """
    # Build audit prompt with all model outputs and merged feature
    merged_gherkin = render_feature(feature, 0, "")  # Render without PR/file metadata

    model_outputs_text = "\n\n".join([
        f"=== {model} output ===\n{output}"
        for model, output in model_outputs.items()
    ])

    audit_prompt = f"""{AUDIT_PROMPT_TEMPLATE}

=== Evidence from code changes ===
{evidence_text}

=== Model outputs ===
{model_outputs_text}

=== Merged Gherkin (to audit) ===
{merged_gherkin}

Respond with a JSON array of findings. Each finding should have:
- "scenario": the scenario name
- "claim": the unsupported claim
- "severity": "high" or "medium"

If all claims are supported, respond with an empty array: []
"""

    # Run audit model
    audit_result = run_model(audit_model, audit_prompt)

    # Save raw audit output
    with open(os.path.join(out_dir, f"{base}.{audit_model}.audit.raw"), "w") as f:
        f.write(audit_result.stdout)

    if audit_result.failure_type != FailureType.SUCCESS:
        # Audit failed - return empty (don't block on audit failure)
        return {f"{audit_model}_findings": []}

    # Parse audit response
    try:
        # Try to extract JSON from response
        output = audit_result.stdout.strip()
        # Handle markdown code blocks
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0].strip()
        elif "```" in output:
            output = output.split("```")[1].split("```")[0].strip()

        findings_data = json.loads(output)
        if not isinstance(findings_data, list):
            return {f"{audit_model}_findings": []}

        # Convert to AuditFinding objects
        findings = []
        for f in findings_data:
            if isinstance(f, dict) and "scenario" in f and "claim" in f:
                findings.append(AuditFinding(
                    scenario=f["scenario"],
                    claim=f["claim"],
                    auditor=audit_model
                ))
        return {f"{audit_model}_findings": findings}
    except (json.JSONDecodeError, KeyError, IndexError):
        return {f"{audit_model}_findings": []}


# ----------------------------- Reasoning Helpers -----------------------------

def _emit_file_reasoning(reasoning_bus: Optional[ReasoningBus], res: Dict[str, Any],
                         threshold: float, audit_model: Optional[str]) -> None:
    """
    Emit reasoning messages for a processed file result.

    This function is called AFTER the decision has been made, to explain why.
    It never influences execution - only documents what happened.
    """
    if reasoning_bus is None:
        return

    file_path = res.get("file", "unknown")
    models_used = res.get("models_used", [])
    model_coverage = res.get("model_coverage", {})
    classified_claims = res.get("classified_claims", [])

    # Emit CLAIM for each model's contribution
    for model in models_used:
        coverage_pct = model_coverage.get(model, 0) * 100
        # Count scenarios attributed to this model
        model_scenarios = [c for c in classified_claims
                          if model in c.get("generating_models", [])]
        if model_scenarios:
            scenario_names = [c.get("scenario", "?")[:40] for c in model_scenarios[:3]]
            reasoning_bus.claim(
                model,
                f"Generated {len(model_scenarios)} scenarios ({coverage_pct:.0f}% coverage): "
                f"{', '.join(scenario_names)}{'...' if len(model_scenarios) > 3 else ''}",
                file=file_path
            )
        elif coverage_pct > 0:
            reasoning_bus.claim(
                model,
                f"Contributed to merged output ({coverage_pct:.0f}% coverage)",
                file=file_path
            )

    # Emit merge decision reasoning
    disagreement = res.get("disagreement", 0.0)

    reasoning_bus.emit_merge_decision(
        file_path=file_path,
        disagreement=disagreement,
        threshold=threshold,
        model_coverage=model_coverage,
        scenario_count=len(classified_claims),
    )

    # Emit classification decisions for each scenario
    for claim in classified_claims:
        reasoning_bus.emit_classification_decision(
            file_path=file_path,
            scenario_name=claim.get("scenario", "unknown"),
            classification=claim.get("classification", "unknown"),
            confidence=claim.get("confidence", 0.0),
            evidence_symbols=claim.get("evidence_symbols", []),
            generating_models=claim.get("generating_models", []),
        )

    # Emit audit decision if audit was performed
    hallucinations = res.get("hallucinations", [])
    model_performance = res.get("model_performance", {})

    if audit_model and audit_model in model_performance:
        audit_perf = model_performance.get(audit_model, {})
        # Approximate needs_review count from classified claims
        trusted_count = sum(1 for c in classified_claims
                          if c.get("classification") in ("grounded", "inferred"))
        needs_review_count = sum(1 for c in classified_claims
                                if c.get("classification") == "contextual")

        findings = [{"scenario": h, "claim": "flagged as hallucination"} for h in hallucinations]

        reasoning_bus.emit_audit_decision(
            file_path=file_path,
            audit_model=audit_model,
            trusted_count=trusted_count,
            needs_review_count=needs_review_count,
            hallucination_count=len(hallucinations),
            findings=findings,
        )

    # Emit final gate decision
    if res.get("failed"):
        failure_reason = res.get("error") or res.get("failure_type") or "unknown reason"
        is_dispute = False
        if hallucinations:
            failure_reason = f"Potential hallucinations detected: {hallucinations}"
        elif disagreement > threshold:
            failure_reason = f"Disagreement {disagreement*100:.1f}% exceeds threshold {threshold*100:.0f}%"
            is_dispute = True  # This is a dispute, not a failure
        reasoning_bus.emit_gate_decision(file_path, passed=False, reason=failure_reason, is_dispute=is_dispute)
    elif res.get("success"):
        reasoning_bus.emit_gate_decision(
            file_path, passed=True,
            reason=f"All checks passed. {len(classified_claims)} scenarios generated."
        )


# ----------------------------- Main -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Gherkin regression tests from PR diffs using LLM consensus.",
        epilog="Use --wizard for interactive configuration."
    )
    ap.add_argument("--repo", required=True, help="GitHub repo (e.g., owner/repo)")
    ap.add_argument("--pr", type=int, required=True, help="Pull request number")
    ap.add_argument("--out", default="out_features", help="Output directory")
    ap.add_argument("--preset", type=str, choices=list(PRESETS.keys()),
                    help="Use a preset configuration: fast (single agent), balanced (two agents), thorough (two agents + audit)")
    ap.add_argument("--sdlc-checks", action="store_true", help="Enable SDLC security checks")
    ap.add_argument("--threshold", type=float, default=DEFAULT_DISAGREEMENT_THRESHOLD,
                    help=f"Disagreement threshold (default: {DEFAULT_DISAGREEMENT_THRESHOLD})")
    ap.add_argument("--timestamp", action="store_true",
                    help="Append timestamp to output directory (e.g., out_features_20260105_093000)")
    ap.add_argument("--audit", action="store_true",
                    help="Enable cross-audit hallucination detection (each model audits the other)")
    ap.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_WORKERS,
                    help=f"Number of parallel workers (default: {DEFAULT_PARALLEL_WORKERS})")
    ap.add_argument("--fail-fast", action="store_true", default=False,
                    help="Stop on first file failure (default: continue and report all)")
    ap.add_argument("--no-disagreement-gate", action="store_true",
                    help="Treat disagreement as advisory (won't fail the run)")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_GENERATION_MODELS),
                    help=f"Comma-separated list of generation models (default: {','.join(DEFAULT_GENERATION_MODELS)})")
    ap.add_argument("--audit-model", type=str, default=DEFAULT_AUDIT_MODEL,
                    help=f"Model to use for auditing (default: {DEFAULT_AUDIT_MODEL})")
    ap.add_argument("--wizard", action="store_true",
                    help="Interactive wizard to configure options")
    ap.add_argument("--db", type=str, default="pr_to_gherkin.db",
                    help="SQLite database path for telemetry logging (default: pr_to_gherkin.db)")
    ap.add_argument("--no-db", action="store_true",
                    help="Disable telemetry database logging")
    ap.add_argument("--db-stats", action="store_true",
                    help="Show database statistics and exit")
    ap.add_argument("--reasoning", type=str, default=None,
                    help="Output file for reasoning transcript (e.g., reasoning.log)")
    ap.add_argument("--reasoning-transcript", action="store_true",
                    help="Print reasoning transcript to stderr at end of run")
    ap.add_argument("--check-models", action="store_true",
                    help="Check agent status (binary, auth, version) and exit")
    ap.add_argument("--validate-auth", action="store_true",
                    help="Validate agent authentication before running (default: off)")
    ap.add_argument("--auto-models", action="store_true",
                    help="Automatically use only available models (skip unavailable)")
    args = ap.parse_args()

    # =========================================================================
    # Preset Handling
    # =========================================================================
    # If a preset is specified, apply its settings (can be overridden by explicit flags)
    # =========================================================================
    if args.preset:
        preset = PRESETS[args.preset]
        sys.stderr.write(f"[info] Using preset '{args.preset}': {preset['description']}\n")

        # Apply preset values only if user didn't explicitly set them
        # Check if --models was explicitly provided (not default)
        if args.models == ",".join(DEFAULT_GENERATION_MODELS):
            args.models = ",".join(preset["models"])

        # Check if --threshold was explicitly provided (not default)
        if args.threshold == DEFAULT_DISAGREEMENT_THRESHOLD:
            args.threshold = preset["threshold"]

        # Check if --parallel was explicitly provided (not default)
        if args.parallel == DEFAULT_PARALLEL_WORKERS:
            args.parallel = preset["parallel"]

        # Enable audit if preset requires it
        if preset.get("audit") and not args.audit:
            args.audit = True
            if "audit_model" in preset:
                args.audit_model = preset["audit_model"]

    # Check models mode - just show status and exit
    if args.check_models:
        print_model_status()
        raise SystemExit(0)

    # Parse and validate models
    args.generation_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # =========================================================================
    # Auto-threshold Adjustment
    # =========================================================================
    # Automatically adjust threshold based on agent count if user hasn't set it
    # Single agent: no disagreement possible
    # Two agents: natural ~50% disagreement
    # Three+ agents: higher disagreement expected
    # =========================================================================
    user_set_threshold = args.threshold != DEFAULT_DISAGREEMENT_THRESHOLD
    if not args.preset and not user_set_threshold:
        num_agents = len(args.generation_models)
        if num_agents == 1:
            # Single agent - no consensus needed, threshold irrelevant
            pass
        elif num_agents == 2:
            args.threshold = 0.55
            sys.stderr.write(f"[info] Auto-adjusted threshold to {args.threshold} for 2-agent consensus\n")
        elif num_agents >= 3:
            args.threshold = 0.70
            sys.stderr.write(f"[info] Auto-adjusted threshold to {args.threshold} for {num_agents}-agent consensus\n")

    # =========================================================================
    # Agent Validation
    # =========================================================================
    # --validate-auth: Full validation (binary + auth + identity)
    # Default: Basic binary availability check only
    # =========================================================================

    agent_validations: Dict[str, AgentValidationResult] = {}

    if args.validate_auth:
        # Full validation: binary, auth, identity
        sys.stderr.write("[info] Validating agent capabilities (auth enabled)...\n")

        all_models_to_validate = list(set(args.generation_models + ([args.audit_model] if args.audit else [])))

        for model in all_models_to_validate:
            try:
                validation = validate_agent(model)
                agent_validations[model] = validation

                if validation.can_participate:
                    sys.stderr.write(
                        f"[agent] {model}: AUTHORIZED "
                        f"(binary: {validation.binary_path}, version: {validation.version_string})\n"
                    )
                else:
                    sys.stderr.write(f"[agent] {model}: FAILED - {validation.failure_reason}\n")
            except Exception as e:
                agent_validations[model] = AgentValidationResult(
                    model=model,
                    binary_available=False,
                    binary_path="unknown",
                    auth_status=False,
                    auth_verified=True,  # We tried to verify
                    version_string="unknown",
                    reported_model="unknown",
                    can_participate=False,
                    failure_reason=f"Validation error: {e}",
                )
                sys.stderr.write(f"[agent] {model}: ERROR - {e}\n")

        # Filter to participating agents
        participating_gen_models = [
            m for m in args.generation_models
            if agent_validations.get(m, AgentValidationResult(
                model=m, binary_available=False, binary_path="", auth_status=False,
                auth_verified=False, version_string="", reported_model="", can_participate=False
            )).can_participate
        ]

        # Handle unavailable generation models
        unavailable_gen = [m for m in args.generation_models if m not in participating_gen_models]
        if unavailable_gen:
            if args.auto_models:
                if not participating_gen_models:
                    for model, spec in MODEL_REGISTRY.items():
                        if spec.role == "generation" and model not in agent_validations:
                            validation = validate_agent(model)
                            agent_validations[model] = validation
                            if validation.can_participate:
                                participating_gen_models.append(model)
                                sys.stderr.write(
                                    f"[agent] {model}: AUTHORIZED (fallback) "
                                    f"(version: {validation.version_string})\n"
                                )

                if participating_gen_models:
                    sys.stderr.write(
                        f"[warn] Models {unavailable_gen} not available/authorized. "
                        f"Using: {participating_gen_models}\n"
                    )
                else:
                    die("No generation models available or authorized. "
                        "Install and authenticate gemini, copilot, or codex CLI.", 2)
            else:
                reasons = [
                    f"{m}: {agent_validations[m].failure_reason}"
                    for m in unavailable_gen if m in agent_validations
                ]
                die(f"Models not available/authorized:\n  " + "\n  ".join(reasons) +
                    "\nUse --auto-models to skip or --check-models to see status.", 2)

        args.generation_models = participating_gen_models

    else:
        # Default: Basic binary availability check only (original behavior)
        # Still populate agent_validations for telemetry (without auth checks)
        sys.stderr.write("[info] Checking model availability...\n")
        available_models = discover_available_models()
        available_list = [m for m, a in available_models.items() if a]
        sys.stderr.write(f"[info] Available models: {', '.join(available_list) if available_list else 'none'}\n")

        # Populate validation info for ALL known agents (for complete audit trail)
        # Explicitly record which agents are selected vs not selected for this run
        selected_models = set(args.generation_models + ([args.audit_model] if args.audit else []))
        for model in MODEL_REGISTRY.keys():
            model_info = get_model_info(model)
            binary_available = check_model_available(model)
            is_selected = model in selected_models

            if not binary_available:
                failure_reason = f"Binary not found"
                can_participate = False
            elif not is_selected:
                failure_reason = f"Not selected for this run"
                can_participate = False
            else:
                failure_reason = None
                can_participate = True

            agent_validations[model] = AgentValidationResult(
                model=model,
                binary_available=binary_available,
                binary_path=model_info["binary_path"],
                auth_status=True,  # Assume auth OK in default mode (not checked)
                auth_verified=False,  # Auth was NOT actually verified in default mode
                version_string=model_info["version_string"],
                reported_model=model_info["reported_model"],
                can_participate=can_participate,
                failure_reason=failure_reason,
            )

        try:
            unavailable_gen = validate_models(args.generation_models, check_availability=True)
            if unavailable_gen:
                if args.auto_models:
                    original = args.generation_models.copy()
                    args.generation_models = [m for m in args.generation_models if m not in unavailable_gen]
                    if not args.generation_models:
                        args.generation_models = [m for m in available_list
                                                  if MODEL_REGISTRY[m].role == "generation"]
                    if not args.generation_models:
                        die("No generation models available. Install gemini, copilot, or codex CLI.", 2)
                    sys.stderr.write(f"[warn] Models {unavailable_gen} not available. Using: {args.generation_models}\n")
                else:
                    die(f"Models not available: {unavailable_gen}. Use --auto-models to skip or --check-models to see status.", 2)

            if args.audit:
                unavailable_audit = validate_models([args.audit_model], check_availability=True)
                if unavailable_audit:
                    if args.auto_models:
                        audit_models = [m for m in available_list if MODEL_REGISTRY[m].role == "audit"]
                        if audit_models:
                            args.audit_model = audit_models[0]
                            sys.stderr.write(f"[warn] Audit model {unavailable_audit[0]} not available. Using: {args.audit_model}\n")
                        else:
                            sys.stderr.write(f"[warn] No audit model available. Using cross-audit between generation models.\n")
                            args.audit_model = None
                    else:
                        die(f"Audit model not available: {args.audit_model}. Use --auto-models to skip or --check-models to see status.", 2)

        except ValueError as e:
            die(str(e), 2)

    # Handle audit model validation (only when --validate-auth is enabled)
    if args.validate_auth and args.audit:
        audit_validation = agent_validations.get(args.audit_model)
        if not audit_validation or not audit_validation.can_participate:
            if args.auto_models:
                # Try to find an available audit model
                for model, spec in MODEL_REGISTRY.items():
                    if spec.role == "audit" and model not in agent_validations:
                        validation = validate_agent(model)
                        agent_validations[model] = validation
                        if validation.can_participate:
                            args.audit_model = model
                            sys.stderr.write(
                                f"[warn] Original audit model not available. "
                                f"Using: {model} (version: {validation.version_string})\n"
                            )
                            break

                # Check if we found an audit model
                audit_validation = agent_validations.get(args.audit_model)
                if not audit_validation or not audit_validation.can_participate:
                    sys.stderr.write(
                        f"[warn] No audit model available/authorized. "
                        f"Using cross-audit between generation models.\n"
                    )
                    args.audit_model = None
            else:
                reason = audit_validation.failure_reason if audit_validation else "Unknown error"
                die(f"Audit model {args.audit_model} not available/authorized: {reason}\n"
                    "Use --auto-models to skip or --check-models to see status.", 2)

    # Ensure we have at least one generation model
    if not args.generation_models:
        die("No generation models specified or available.", 2)

    # Database stats mode
    if args.db_stats:
        if args.no_db:
            die("--db-stats cannot be used with --no-db", 2)
        if not os.path.exists(args.db):
            die(f"Database not found: {args.db}", 2)

        db = TelemetryDB(args.db)
        print("\n=== Gherkin Telemetry Database Statistics ===\n")

        # Schema info
        schema_version = db.get_schema_version()
        print(f"Database: {args.db}")
        print(f"Schema version: {schema_version} (script: {TelemetryDB.SCHEMA_VERSION})")
        print()

        # Run history
        runs = db.get_run_history(limit=10)
        print(f"Recent runs ({len(runs)} shown):")
        for run in runs:
            gen_models = json.loads(run["generation_models"])
            print(f"  {run['timestamp'][:19]} | {run['repo']} PR#{run['pr_number']} | "
                  f"{run['files_succeeded']}/{run['files_total']} succeeded | "
                  f"models: {','.join(gen_models)}")
        print()

        # Model performance
        model_stats = db.get_model_stats()
        print("Model performance summary:")
        for stat in model_stats:
            if stat["role"] == "generation":
                print(f"  {stat['model_name']} (generation):")
                print(f"    Files processed: {stat['total_files']}")
                print(f"    Total scenarios: {stat['total_scenarios'] or 0}")
                print(f"    Avg confidence: {stat['overall_avg_confidence']:.3f}" if stat['overall_avg_confidence'] else "    Avg confidence: N/A")
                print(f"    Grounded ratio: {stat['avg_grounded_ratio']:.3f}" if stat['avg_grounded_ratio'] else "    Grounded ratio: N/A")
            elif stat["role"] == "audit":
                print(f"  {stat['model_name']} (audit):")
                print(f"    Files audited: {stat['total_files']}")
                print(f"    Findings: {stat['total_findings'] or 0}")
                print(f"    Hallucinations flagged: {stat['total_hallucinations'] or 0}")
        print()

        # Layer stats
        layer_stats = db.get_layer_stats()
        print("Scenario coverage by layer:")
        current_layer = None
        for stat in layer_stats:
            if stat["layer"] != current_layer:
                current_layer = stat["layer"]
                print(f"  {current_layer}:")
            print(f"    {stat['classification']}: {stat['count']} scenarios "
                  f"(avg confidence: {stat['avg_confidence']:.3f})" if stat['avg_confidence'] else
                  f"    {stat['classification']}: {stat['count']} scenarios")
        print()

        # Hallucination rates
        hall_rates = db.get_hallucination_rate_by_model()
        if hall_rates:
            print("Hallucination rates by generating model:")
            for rate in hall_rates:
                print(f"  {rate['model_name']}: {rate['hallucination_rate']*100:.1f}% "
                      f"({rate['flagged_scenarios']}/{rate['total_scenarios']} flagged)")

        db.close()
        raise SystemExit(0)

    # Initialize telemetry database (on by default)
    global _telemetry_db, _reasoning_bus
    telemetry_db = None
    reasoning_bus = None
    run_id = None
    run_start_time = datetime.now()

    if not args.no_db:
        telemetry_db = TelemetryDB(args.db)
        _telemetry_db = telemetry_db
        sys.stderr.write(f"[info] Telemetry logging to: {args.db}\n")

    # Initialize reasoning bus (for explaining decisions)
    reasoning_file = args.reasoning
    if reasoning_file is None and args.timestamp:
        # Auto-generate reasoning file in output dir
        pass  # Will be set after out_dir is determined

    # Wizard mode: interactive configuration
    if args.wizard:
        config = run_wizard(args.repo, args.pr)
        print_wizard_summary(config)

        # Show the generated CLI command
        cli_cmd = config_to_cli(config)
        print("\n  Generated command:")
        print("  " + "-" * 56)
        print(f"  {' '.join(cli_cmd)}")
        print("  " + "-" * 56)

        if not prompt_yn("\n  Run this command?", default=True):
            print("\n  Wizard complete. Copy the command above to run later.")
            raise SystemExit(0)

        # Apply wizard config to args
        args.audit = config["audit"]
        args.parallel = config["parallel"]
        args.threshold = config["threshold"]
        args.fail_fast = config["fail_fast"]
        args.timestamp = config["timestamp"]
        args.sdlc_checks = config["sdlc_checks"]
        args.no_disagreement_gate = config.get("no_disagreement_gate", False)
        print("\n" + "=" * 60)
        print("  Starting execution...")
        print("=" * 60 + "\n")

    # Append timestamp to output directory if requested
    out_dir = args.out
    if args.timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{args.out}_{ts}"

    ensure_dir(out_dir)

    # LangChain graph - also extract evidence for auditing
    def analyze_patch(d: dict) -> dict:
        analyzer = PatchAnalyzer(d["patch"])
        return {
            **d,
            "files": analyzer.summarize(),
            "evidence": analyzer.extract_evidence(),
        }

    summarize = RunnableLambda(analyze_patch)

    distill = RunnableLambda(lambda d: {
        **d,
        "hints": format_hints(distill_hints(d["context"]))
    })

    def prompt_builder(d: dict) -> List[dict]:
        prompts = []
        for path, changes in d["files"].items():
            prompt = f"""Generate REGRESSION Gherkin test scenarios for the following code changes.

File: {path}
Changes: {changes}

Context from PR:
{d['hints']}

Rules:
- These are regression tests: existing externally observable behavior must not change.
- If there are shared preconditions, express them using a single Background: section.
- Do not repeat Background steps inside scenarios.
- Output ONLY valid Gherkin.
- Start with "Feature:".
- Include one or more "Scenario:" blocks.
- Use Given/When/Then steps only.
- No explanations, no markdown."""
            prompts.append({
                "file": path,
                "prompt": prompt,
                "evidence": d.get("evidence", {}),
                "context": d.get("context", {}),
            })
        return prompts

    build_prompts = RunnableLambda(prompt_builder)

    chain = gh_pr_bundle | summarize | distill | build_prompts

    bundle = chain.invoke({"repo": args.repo, "pr": args.pr})

    num_files = len(bundle)
    sys.stderr.write(f"[info] Processing {num_files} files with {args.parallel} worker(s)\n")

    # Determine reasoning output path
    reasoning_output = args.reasoning or os.path.join(out_dir, "reasoning.log")

    # Build resolved configuration for persistence
    resolved_config = {
        "models": args.generation_models,
        "threshold": args.threshold,
        "parallel": args.parallel,
        "audit": args.audit,
        "audit_model": args.audit_model if args.audit else None,
        "preset": getattr(args, 'preset', None),
        "validate_auth": args.validate_auth,
        "auto_models": args.auto_models,
        "fail_fast": args.fail_fast,
        "no_disagreement_gate": args.no_disagreement_gate,
    }

    # Determine run classification based on configuration
    num_agents = len(args.generation_models)
    if num_agents == 1 and not args.audit:
        run_classification = "baseline"
    elif num_agents >= 2 and not args.audit:
        run_classification = "coverage_exploration"
    elif args.audit:
        run_classification = "audit_strict"
    else:
        run_classification = "mixed"

    # Start telemetry run
    if telemetry_db:
        run_id = telemetry_db.start_run(
            repo=args.repo,
            pr_number=args.pr,
            threshold=args.threshold,
            audit_enabled=args.audit,
            generation_models=args.generation_models,
            audit_model=args.audit_model if args.audit else None,
            output_dir=out_dir,
            reasoning_file=reasoning_output,
            script_version=SCRIPT_VERSION,
            preset_name=getattr(args, 'preset', None),
            resolved_config=resolved_config,
            run_classification=run_classification,
        )
        sys.stderr.write(f"[telemetry] Run started: {run_id[:8]}...\n")

    # Initialize reasoning bus with run_id
    reasoning_bus = ReasoningBus(run_id=run_id, output_file=reasoning_output)
    _reasoning_bus = reasoning_bus
    sys.stderr.write(f"[reasoning] Transcript logging to: {reasoning_output}\n")

    # Emit run start decision
    reasoning_bus.decision(
        f"Starting PR #{args.pr} from {args.repo}. "
        f"Processing {num_files} files with models: {', '.join(args.generation_models)}. "
        f"Audit: {'enabled (' + args.audit_model + ')' if args.audit else 'disabled'}."
    )

    # Emit DECISION reasoning for each agent validation and log to telemetry
    for model, validation in agent_validations.items():
        if validation.can_participate:
            reasoning_bus.decision(
                f"Agent '{model}' authorized. "
                f"Reported model: {validation.reported_model}. "
                f"Binary: {validation.binary_path}."
            )
        else:
            reasoning_bus.decision(
                f"Agent '{model}' DISABLED: {validation.failure_reason}"
            )

        # Log validation to telemetry database
        if telemetry_db and run_id:
            try:
                telemetry_db.log_agent_validation(run_id, validation)
            except Exception as e:
                sys.stderr.write(f"[telemetry] Warning: Failed to log validation for {model}: {e}\n")

    results = []
    any_fail = False

    # Get no_disagreement_gate safely (may not exist if not using wizard)
    no_disagreement_gate = getattr(args, 'no_disagreement_gate', False)

    if args.parallel <= 1:
        # Sequential processing
        for i, item in enumerate(bundle):
            sys.stderr.write(f"[{i+1}/{num_files}] {item['file']}\n")
            res = process_single_file(
                item, out_dir, args.pr, args.threshold,
                args.sdlc_checks, args.audit, no_disagreement_gate,
                args.generation_models, args.audit_model
            )
            results.append(res)

            # Emit reasoning for this file
            _emit_file_reasoning(reasoning_bus, res, args.threshold, args.audit_model if args.audit else None)

            if res["failed"]:
                any_fail = True
                if args.fail_fast:
                    sys.stderr.write(f"[error] Fail-fast: stopping due to failure in {item['file']}\n")
                    break
    else:
        # Parallel processing with bounded worker pool
        max_workers = min(args.parallel, num_files, os.cpu_count() or 4)
        sys.stderr.write(f"[info] Using {max_workers} parallel workers\n")
        sys.stderr.write(f"[info] Generation models: {', '.join(args.generation_models)}\n")
        if args.audit:
            sys.stderr.write(f"[info] Audit model: {args.audit_model}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_single_file, item, out_dir, args.pr, args.threshold,
                    args.sdlc_checks, args.audit, no_disagreement_gate,
                    args.generation_models, args.audit_model
                ): item["file"]
                for item in bundle
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed += 1

                try:
                    res = future.result()
                    results.append(res)
                    status = "✓" if res["success"] and not res["failed"] else "✗"
                    sys.stderr.write(f"[{completed}/{num_files}] {status} {file_path}\n")

                    if res["failed"]:
                        any_fail = True
                        if res["error"]:
                            sys.stderr.write(f"    Error: {res['error']}\n")
                        if args.fail_fast:
                            sys.stderr.write(f"[error] Fail-fast: cancelling remaining tasks\n")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                except Exception as e:
                    any_fail = True
                    sys.stderr.write(f"[{completed}/{num_files}] ✗ {file_path}: {e}\n")
                    results.append({"file": file_path, "success": False, "error": str(e)})
                    if args.fail_fast:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

        # Emit reasoning for all parallel results (after collection)
        for res in results:
            _emit_file_reasoning(reasoning_bus, res, args.threshold, args.audit_model if args.audit else None)

    # Summary - distinguish between disputed (high disagreement) and failed (actual errors)
    succeeded = sum(1 for r in results if r.get("success") and not r.get("failed"))

    # Disputed: generated output but models disagreed above threshold
    disputed = sum(1 for r in results
                   if r.get("failed")
                   and r.get("disagreement", 0) > args.threshold
                   and not r.get("error"))

    # Failed: actual errors (parse, timeout, API, etc.)
    failed = sum(1 for r in results if r.get("failed")) - disputed

    # Log telemetry for all processed files
    if telemetry_db and run_id:
        for res in results:
            try:
                # Determine status: success, disputed, or failed
                if res.get("success") and not res.get("failed"):
                    status = OutcomeStatus.SUCCESS
                elif res.get("failed") and res.get("disagreement", 0) > args.threshold and not res.get("error"):
                    status = OutcomeStatus.DISPUTED
                else:
                    status = OutcomeStatus.FAILED

                # Determine dispute reason if status is disputed
                dispute_reason = None
                if status == OutcomeStatus.DISPUTED:
                    # Analyze model coverage to determine dispute type
                    model_coverage = res.get("model_coverage", {})
                    claims = res.get("classified_claims", [])
                    if model_coverage:
                        coverage_values = list(model_coverage.values())
                        # If all models have similar coverage, it's coverage_diversity
                        # If coverage varies significantly, it's likely contradiction
                        if coverage_values and max(coverage_values) - min(coverage_values) < 0.3:
                            dispute_reason = "coverage_diversity"
                        else:
                            dispute_reason = "contradiction"
                    else:
                        dispute_reason = "mixed"

                # Log file result
                file_id = telemetry_db.log_file_result(
                    run_id=run_id,
                    file_path=res.get("file", "unknown"),
                    status=status,
                    failure_type=res.get("failure_type"),
                    error_message=res.get("error"),
                    disagreement=res.get("disagreement"),
                    dispute_reason=dispute_reason,
                    domain=detect_domain(res.get("file", "")),
                )

                # Log model outputs (from raw files in output dir)
                for model in res.get("models_used", []):
                    raw_path = os.path.join(out_dir, f"{safe_name(res.get('file', ''))}.{model}.raw")
                    raw_output = ""
                    if os.path.exists(raw_path):
                        with open(raw_path, "r") as f:
                            raw_output = f.read()
                    telemetry_db.log_model_output(
                        file_id=file_id,
                        model_name=model,
                        role="generation",
                        raw_output=raw_output,
                        cleaned_output=clean_llm_output(raw_output) if raw_output else None,
                        parse_success=res.get("success", False),
                        failure_type=res.get("failure_type") if model in str(res.get("error", "")) else None,
                    )

                # Log scenarios and their classifications
                for claim in res.get("classified_claims", []):
                    telemetry_db.log_scenario(
                        file_id=file_id,
                        scenario_name=claim["scenario"],
                        steps=[],  # Steps are in the feature file
                        tags=[],
                        classification=claim["classification"],
                        confidence=claim["confidence"],
                        layer=claim["layer"],
                        evidence_symbols=claim.get("evidence_symbols", []),
                        generating_models=claim.get("generating_models", []),
                    )

                # Log model performance metrics
                for model, metrics in res.get("model_performance", {}).items():
                    telemetry_db.log_model_performance(
                        file_id=file_id,
                        run_id=run_id,
                        model_name=model,
                        role=metrics.get("role", "generation"),
                        metrics=metrics,
                    )

                # Log audit findings (hallucinations)
                for hallucination in res.get("hallucinations", []):
                    telemetry_db.log_audit_finding(
                        file_id=file_id,
                        scenario_name=hallucination,
                        claim="flagged as hallucination",
                        auditor_model=args.audit_model if args.audit else "cross-audit",
                        severity="high",
                    )

            except Exception as e:
                sys.stderr.write(f"[telemetry] Warning: Failed to log {res.get('file')}: {e}\n")

        # Finish the run
        run_duration = (datetime.now() - run_start_time).total_seconds()
        telemetry_db.finish_run(
            run_id=run_id,
            files_total=len(results),
            files_succeeded=succeeded,
            files_failed=failed,
            duration_seconds=run_duration,
            files_disputed=disputed,
        )
        telemetry_db.close()
        sys.stderr.write(f"[telemetry] Run completed: {run_id[:8]}... ({run_duration:.1f}s)\n")

    # Finalize reasoning bus
    if reasoning_bus:
        # Emit final run summary decision
        summary_parts = [f"{succeeded} succeeded"]
        if disputed > 0:
            summary_parts.append(f"{disputed} disputed")
        if failed > 0:
            summary_parts.append(f"{failed} failed")
        reasoning_bus.decision(
            f"Run completed: {', '.join(summary_parts)} out of {len(results)} files. "
            f"Total decisions: {len(reasoning_bus.get_decisions())}. "
            f"Total challenges: {len(reasoning_bus.get_challenges())}."
        )

        # Persist reasoning to database
        if telemetry_db and not args.no_db:
            # Reopen DB for reasoning persistence (was closed above)
            reasoning_db = TelemetryDB(args.db)
            persisted = reasoning_bus.persist_to_db(reasoning_db)
            reasoning_db.close()
            sys.stderr.write(f"[reasoning] Persisted {persisted} messages to database\n")

        # Print transcript if requested
        if args.reasoning_transcript:
            sys.stderr.write("\n" + "=" * 60 + "\n")
            sys.stderr.write("  REASONING TRANSCRIPT\n")
            sys.stderr.write("=" * 60 + "\n")
            sys.stderr.write(reasoning_bus.get_transcript() + "\n")
            sys.stderr.write("=" * 60 + "\n\n")

        reasoning_bus.close()
        sys.stderr.write(f"[reasoning] Transcript saved to: {reasoning_output}\n")

    # Final summary
    summary_parts = [f"{succeeded} succeeded"]
    if disputed > 0:
        summary_parts.append(f"{disputed} disputed")
    if failed > 0:
        summary_parts.append(f"{failed} failed")
    sys.stderr.write(f"\n[summary] {', '.join(summary_parts)} out of {len(results)} files\n")
    sys.stderr.write(f"[output] {out_dir}\n")

    # Exit code: 0 = all success, 1 = disputes only, 4 = actual failures
    if failed > 0:
        raise SystemExit(4)
    elif disputed > 0:
        raise SystemExit(1)
    else:
        raise SystemExit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        die(str(exc), 3)