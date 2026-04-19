"""
Smoke test — run this while the Flask server is running.
Usage: python test_predict.py
"""
import json

import requests

BASE_URL = "http://localhost:9696"


def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    print(f"✅ /health  →  {data}")


def test_predict():
    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 2.5,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=ride)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_duration_minutes" in data
    duration = data["predicted_duration_minutes"]
    assert 1 <= duration <= 120, f"Implausible prediction: {duration}"
    print(f"✅ /predict →  {data}")


def test_missing_fields():
    resp = requests.post(f"{BASE_URL}/predict", json={"PULocationID": 1})
    assert resp.status_code == 400
    print(f"✅ missing fields validation →  {resp.json()}")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_health()
    test_predict()
    test_missing_fields()
    print("\n✅ All tests passed!")
