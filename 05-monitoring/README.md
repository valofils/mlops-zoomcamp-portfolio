# Module 5: Model Monitoring

## What This Module Covers

- Why models degrade over time
- Data drift vs. concept drift vs. model degradation
- Building monitoring dashboards
- Alerting on drift detection

## Tools

- **Evidently AI** — data and model drift reports
- **Grafana** — dashboards and visualization
- **Prometheus** — metrics collection
- **MongoDB** — storing monitoring results

## Structure

| Folder | Contents |
|--------|----------|
| `dashboards/` | Grafana dashboard configs |
| `scripts/` | Monitoring and reporting scripts |
| `data/` | Reference datasets for drift comparison |

## Key Concepts

- **Data drift**: input feature distribution has changed
- **Concept drift**: relationship between features and target has changed
- **Reference dataset**: baseline data the model was trained on
- **Current dataset**: recent production data to compare against

## Notes & Learnings

_Add your notes here as you work through the module._
