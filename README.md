# Clinical Trial Agent

Multi-agent system for clinical trial matching, eligibility reasoning, and biomedical data extraction.

## Overview

Clinical Trial Agent leverages stateful, modular agents to automate the process of matching patients to clinical trials, parsing eligibility criteria, extracting patient data, and normalizing biomedical terminology. Built with LangGraph, LangChain, and modern Python tools, it is designed for extensibility, explainability, and integration with clinical data sources.

## Features

- **Automated Clinical Trial Matching**: Multi-agent workflow for matching patient profiles to trial eligibility.
- **Eligibility Reasoning**: Structured verdicts (MEETS, FAILS, UNCERTAIN) with explainable logic.
- **Criteria Parsing**: Converts free-text trial criteria into atomic, assessable statements.
- **Patient Data Extraction**: Extracts structured data from clinical narratives.
- **Terminology Normalization**: Maps conditions and drugs to canonical forms (MeSH, ICD-10, INN).
- **Extensible Agent Architecture**: Easily add new agents, tools, or data sources.

## Installation

**Requirements:** Python 3.11+, PostgreSQL (for memory/cache), API keys for LLM providers (OpenAI, Gemini, etc.)

1. Clone the repository:
 ```bash
 git clone https://github.com/Chrisolande/clinical_trial_agent.git
 cd clinical_trial_agent
 ```

2. Create and activate a virtual environment:
 ```bash
 python3.11 -m venv .venv
 source .venv/bin/activate
 ```

3. Install dependencies:
 ```bash
 pip install -e .
 ```

4. Configure environment variables:
 - Copy `.env.example` to `.env` and fill in required API keys and database URIs.

## Usage

Run the main CLI or integrate as a Python module:

```bash
clinical-trial-agent --help
```

Or use the Python API:

```python
from agents.trial_search import search_trials
results = search_trials(condition="lung cancer", intervention="pembrolizumab")
```

## Project Structure

```
clinical_trial_agent/
├── agents/         # Core agent modules (parsing, reasoning, search, etc.)
├── models/         # Data models (criteria, patient, terminology)
├── prompts/        # Prompt templates for LLM agents
├── subagents/      # Specialized subagent workflows (eligibility, retrieval, synthesis)
├── tools/          # Utility and integration tools (cache, validation, sanitizer)
├── clinical_trials.py  # ClinicalTrials.gov API integration
├── config.py           # Environment and settings management
├── memory.py           # Memory/cache layer
├── logging_config.py   # Logging setup
├── validate_env.py     # Environment validation
├── pyproject.toml      # Project metadata and dependencies
```

## Contributing

Pull requests and issues are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) if available, or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Author: Chris Olande
Email: <olandechris@gmail.com>
