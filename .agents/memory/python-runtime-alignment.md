---
name: Python environment alignment
description: Keep package installs in the same Python version used by the Replit workflow.
---

When a Replit project has a fixed Python module in `.replit`, keep a matching `.python-version` so uv resolves dependencies for that interpreter. A broad `requires-python` range alone may select a different system Python.

**Why:** Package installation selected Python 3.12 while the workflow ran Python 3.11, leaving Streamlit installed outside the workflow's import path.

**How to apply:** When Python imports fail after a successful dependency installation, compare the workflow interpreter and uv's selected interpreter before reinstalling packages.
