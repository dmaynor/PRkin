# PRkin - PR to Gherkin Consensus Pipeline

A LangChain-enabled pipeline that generates Gherkin regression tests from GitHub Pull Request diffs using multi-agent LLM consensus.

## What It Does

This tool:

1. **Fetches PR diffs** from GitHub repositories
2. **Generates Gherkin test scenarios** using multiple LLM agents (Gemini, GitHub Copilot, OpenAI Codex, Claude)
3. **Builds consensus** across agents to reduce hallucinations and improve test quality
4. **Optionally audits** generated scenarios for hallucination detection
5. **Produces CI-ready output** with disagreement gates and detailed telemetry

The core idea: when multiple independent LLMs agree on a test scenario, it's more likely to be grounded in the actual code changes.

## Requirements

### Python Dependencies

```bash
pip install unidiff langchain
```

### LLM CLI Tools

You need at least one of the following CLI tools installed and authenticated:

| Agent | Install | Authenticate |
|-------|---------|--------------|
| **Gemini** | `brew install gemini-cli` or [manual install](https://github.com/google/gemini-cli) | `gemini auth login` |
| **GitHub Copilot** | `gh extension install github/gh-copilot` | `gh auth login` |
| **OpenAI Codex** | `npm install -g @openai/codex` | `codex auth` or set `OPENAI_API_KEY` |
| **Claude** | `npm install -g @anthropic-ai/claude-code` | `claude auth login` |

### GitHub CLI

Required for fetching PR data:

```bash
brew install gh
gh auth login
```

## Quick Start

### Check Available Agents

```bash
python3 prkin.py --repo owner/repo --pr 123 --check-models
```

Output:
```
Agent Capability Status:
======================================================================
Agent        Role         Binary     Auth       Version
----------------------------------------------------------------------
  gemini     generation   ✓          ✓          0.22.5
  copilot    generation   ✓          ✓          0.0.374
  codex      generation   ✓          ✓          codex-cli 0.77.0
  claude     audit        ✓          ✓          2.0.76 (Claude Code)
----------------------------------------------------------------------
  4/4 agents ready to participate
```

### Basic Usage

```bash
# Generate tests using default settings (copilot, parallel=2)
python3 prkin.py \
  --repo owner/repo \
  --pr 123
```

### Using Presets

Presets provide optimized configurations based on telemetry analysis:

```bash
# Fast: Single agent (copilot), fastest execution
python3 prkin.py \
  --repo owner/repo --pr 123 \
  --preset fast

# Balanced: Two agents (copilot + codex), complementary coverage
python3 prkin.py \
  --repo owner/repo --pr 123 \
  --preset balanced

# Thorough: Two agents + Claude audit for hallucination detection
python3 prkin.py \
  --repo owner/repo --pr 123 \
  --preset thorough
```

## Configuration Options

### Agent Selection

```bash
# Use specific models
--models copilot,gemini,codex

# Auto-select available models (skip unavailable)
--auto-models

# Validate authentication before running
--validate-auth
```

### Consensus Settings

```bash
# Set disagreement threshold (default: 0.50)
# Files with disagreement above this are marked "disputed"
--threshold 0.55

# Treat disagreement as advisory (won't fail the run)
--no-disagreement-gate
```

### Audit Mode

```bash
# Enable hallucination detection audit
--audit

# Specify audit model (default: claude)
--audit-model claude
```

### Execution Control

```bash
# Number of parallel workers (default: 2)
--parallel 4

# Stop on first failure
--fail-fast

# Append timestamp to output directory
--timestamp
```

## Output Structure

### Directory Layout

```
out_features_20260105_143000/
├── file1.feature              # Merged Gherkin scenarios
├── file1.copilot.raw          # Raw output from copilot
├── file1.codex.raw            # Raw output from codex
├── file2.feature
├── ...
└── reasoning.log              # Decision transcript
```

### Reasoning Transcript

The reasoning log captures all decisions made during the run:

```
[2026-01-05T14:30:00] [DECISION] [system] Starting PR #123 from owner/repo...
[2026-01-05T14:30:01] [DECISION] [system] Agent 'copilot' authorized. Binary: /usr/local/bin/copilot
[2026-01-05T14:30:05] [CLAIM] [copilot] Generated 5 scenarios (67% coverage)...
[2026-01-05T14:30:08] [CLAIM] [codex] Generated 4 scenarios (55% coverage)...
[2026-01-05T14:30:10] [DECISION] [system] Merged 6 scenarios with 42% disagreement...
```

### Custom Reasoning Output

```bash
# Write reasoning to specific file
--reasoning /path/to/reasoning.log

# Print transcript to stderr at end of run
--reasoning-transcript
```

## Telemetry Database

All runs are logged to a SQLite database for analysis.

### View Statistics

```bash
python3 prkin.py \
  --repo owner/repo --pr 123 \
  --db-stats
```

Output:
```
=== Gherkin Telemetry Database Statistics ===

Database: pr_to_gherkin.db
Schema version: 5 (script: 5)

Recent runs (10 shown):
  2026-01-05T18:02:23 | owner/repo PR#123 | 5/5 succeeded | models: copilot
  2026-01-05T17:45:10 | owner/repo PR#123 | 3/5 succeeded | models: copilot,codex

Model performance summary:
  copilot (generation):
    Files processed: 15
    Total scenarios: 102
    Avg confidence: 0.248
  codex (generation):
    Files processed: 3
    Total scenarios: 15
    Avg confidence: 0.250
```

### Database Options

```bash
# Use custom database path
--db /path/to/telemetry.db

# Disable database logging
--no-db
```

### Database Schema

The database tracks:

| Table | Purpose |
|-------|---------|
| `runs` | Pipeline executions with config, timing, outcomes |
| `file_results` | Per-file status, disagreement, dispute reasons |
| `model_outputs` | Raw and cleaned output from each agent |
| `scenarios` | Individual test scenarios with classification |
| `agent_validations` | Agent capability checks per run |
| `reasoning_messages` | Full decision audit trail |
| `model_performance` | Per-model metrics and quality scores |
| `audit_findings` | Hallucinations flagged by auditor |

### Querying the Database

```bash
# View recent runs
sqlite3 pr_to_gherkin.db "
  SELECT timestamp, repo, pr_number,
         files_succeeded, files_disputed, files_failed,
         run_classification
  FROM runs
  ORDER BY timestamp DESC
  LIMIT 10;
"

# Analyze disagreement by model pair
sqlite3 pr_to_gherkin.db "
  SELECT generation_models,
         AVG(disagreement) as avg_disagreement,
         COUNT(*) as file_count
  FROM file_results f
  JOIN runs r ON f.run_id = r.run_id
  GROUP BY generation_models;
"

# Find files with coverage diversity disputes
sqlite3 pr_to_gherkin.db "
  SELECT file_path, disagreement, dispute_reason
  FROM file_results
  WHERE dispute_reason = 'coverage_diversity';
"

# Check agent auth verification status
sqlite3 pr_to_gherkin.db "
  SELECT agent_name,
         SUM(auth_verified) as verified_count,
         SUM(1 - auth_verified) as assumed_count
  FROM agent_validations
  GROUP BY agent_name;
"
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All files succeeded |
| 1 | Some files disputed (consensus issues) |
| 4 | Actual failures (parse errors, API errors, etc.) |

## Examples

### CI Integration

```bash
# Run in CI with strict mode
python3 prkin.py \
  --repo $GITHUB_REPOSITORY \
  --pr $PR_NUMBER \
  --preset balanced \
  --timestamp \
  --fail-fast

# Check exit code
if [ $? -eq 0 ]; then
  echo "All tests generated successfully"
elif [ $? -eq 1 ]; then
  echo "Some files had disagreement - review needed"
else
  echo "Generation failed"
  exit 1
fi
```

### Interactive Exploration

```bash
# Use wizard for interactive configuration
python3 prkin.py \
  --repo owner/repo --pr 123 \
  --wizard
```

### Full Audit Run

```bash
# Maximum coverage with audit and detailed logging
python3 prkin.py \
  --repo owner/repo \
  --pr 123 \
  --models copilot,codex,gemini \
  --audit \
  --audit-model claude \
  --threshold 0.65 \
  --parallel 4 \
  --timestamp \
  --reasoning-transcript
```

### Comparing Agent Combinations

```bash
# Run multiple configurations with timestamps
for preset in fast balanced thorough; do
  python3 prkin.py \
    --repo owner/repo --pr 123 \
    --preset $preset \
    --timestamp
done

# Compare results in database
python3 prkin.py \
  --repo owner/repo --pr 123 \
  --db-stats
```

## Understanding Results

### Outcome Status

- **Success**: All agents agreed, scenarios merged successfully
- **Disputed**: Disagreement exceeded threshold but no errors (needs review)
- **Failed**: Actual errors occurred (API, parsing, timeout)

### Dispute Reasons

When files are disputed, the reason is categorized:

- **coverage_diversity**: Agents produced different but non-contradictory scenarios
- **contradiction**: Agents made conflicting claims about the same behavior
- **mixed**: Unable to determine specific cause

### Run Classifications

Runs are automatically tagged:

- **baseline**: Single agent, no audit (fast, no consensus)
- **coverage_exploration**: Multiple agents, no audit (consensus building)
- **audit_strict**: Any config with audit enabled (hallucination detection)

## Troubleshooting

### Agent Not Found

```
Agent 'gemini' is not installed or not on PATH.
```

Install the missing CLI tool and ensure it's on your PATH.

### Authentication Failed

```
Agent 'copilot' is not authenticated. Please authenticate first.
```

Run the authentication command for that agent (see Requirements section).

### Quota Exhausted

```
You have exhausted your daily quota on this model.
```

The agent is authenticated but rate-limited. Wait or use a different agent.

### High Disagreement

If most runs show high disagreement:
1. Try raising `--threshold` to 0.60-0.70
2. Use `--no-disagreement-gate` to treat as advisory
3. Review the reasoning log to understand differences

## Version

Current version: **1.5.0** (Schema v5)

Check version:
```bash
python3 -c "import prkin; print(prkin.SCRIPT_VERSION)"
```
