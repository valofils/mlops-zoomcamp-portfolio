# MLOps Zoomcamp Portfolio

End-to-end MLOps project portfolio built while completing the
[DataTalks.Club MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).
Each module introduces a new layer of the ML production stack.

## Modules

| Module | Topic | Tools | Status |
|--------|-------|-------|--------|
| [01-intro](./01-intro/) | ML in production, environment setup | Python, Docker | 🔄 In progress |
| [02-experiment-tracking](./02-experiment-tracking/) | Experiment tracking & model registry | MLflow | ⏳ Upcoming |
| [03-orchestration](./03-orchestration/) | ML pipelines & workflow orchestration | Prefect | ⏳ Upcoming |
| [04-deployment](./04-deployment/) | Web service, streaming & batch deployment | Flask, AWS | ⏳ Upcoming |
| [05-monitoring](./05-monitoring/) | Model & data drift monitoring | Evidently, Grafana | ⏳ Upcoming |
| [06-best-practices](./06-best-practices/) | Testing, CI/CD, IaC | pytest, GitHub Actions, Terraform | ⏳ Upcoming |
| [07-project](./07-project/) | End-to-end capstone project | All of the above | ⏳ Upcoming |

## Stack

- **Experiment Tracking**: MLflow
- **Orchestration**: Prefect
- **Deployment**: Flask, FastAPI, AWS Lambda
- **Monitoring**: Evidently AI, Grafana, Prometheus
- **CI/CD**: GitHub Actions
- **Infrastructure**: Docker, Terraform
- **Environment**: GitHub Codespaces

## Setup

```bash
git clone https://github.com/valofils/mlops-zoomcamp-portfolio
cd mlops-zoomcamp-portfolio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See each module's README for specific setup instructions.
