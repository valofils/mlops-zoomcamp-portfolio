# Module 7: Final Project

## Overview

End-to-end MLOps project integrating all course concepts.
Dataset TBD — will use a more distinctive problem domain than the Taxi dataset.

## Checklist

### Problem description
- [ ] Problem clearly described, solution is valuable

### Experiment tracking & model registry
- [ ] MLflow used for experiment tracking
- [ ] Models registered in a model registry

### Workflow orchestration
- [ ] Fully deployed orchestration pipeline

### Model deployment
- [ ] Model deployed as web service, streaming, or batch

### Model monitoring
- [ ] Dashboard for monitoring model performance

### Reproducibility
- [ ] Clear setup instructions, dependencies pinned, data obtainable

### Best practices
- [ ] Unit tests
- [ ] Integration test
- [ ] Linter and/or formatter
- [ ] Makefile
- [ ] Pre-commit hooks
- [ ] CI/CD pipeline

## Structure

| Folder | Contents |
|--------|----------|
| `src/` | Source code — training, prediction, utils |
| `notebooks/` | EDA and experimentation |
| `data/` | Processed data (raw excluded via .gitignore) |
| `tests/` | Unit and integration tests |
| `infrastructure/` | Terraform and Docker configs |

## Notes & Learnings

_Add your notes here as you work through the project._
