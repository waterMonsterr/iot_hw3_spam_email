# OpenSpec Combined Change Proposal — 0001 (MQTT simulator) & 0005 (Spam classification baseline)

author: GitHub Copilot  
date: 2025-10-29  
status: draft

This combined proposal aligns the two phases with the repository policy (openspec/project.md), AI-assisted workflow (ai-coding-cli), and includes a Streamlit demo for presentation.

---

## Phase A — 0001: MQTT sensor simulator + REST /status endpoint

title: MQTT simulator + REST /status endpoint  
status: draft

scope (canonical files)
- openspec/tools/mqtt_simulator.py
- openspec/requirements-tools.txt
- tests/python/test_mqtt_simulator.py
- docker-compose.mosquitto.yml (optional for CI)

design highlights
- Publish JSON to topic: iot/iot__hw3/sensor/<id>
- Default broker: localhost:1883. CLI: --broker --port --rate --topic --id
- Optional FastAPI /status endpoint to return last message
- Minimal base implementation uses paho-mqtt; FastAPI optional

AI-assisted implementation steps
1. Create openspec change file (openspec/changes/0001-mqtt-simulator.md).
2. Run: `ai-coding-cli scaffold --spec openspec/changes/0001-mqtt-simulator.md`
3. Review generated files and tests.
4. Run: `ai-coding-cli verify --spec openspec/changes/0001-mqtt-simulator.md` (lint + pytest + optional integration smoke test)

testing & CI
- Unit tests in tests/python/
- Optional integration job in CI: start docker-compose.mosquitto.yml → run pytest tests/python/test_mqtt_simulator.py → tear down

success criteria
- Simulator publishes at configured rate; /status returns last payload; relevant tests pass

---

## Phase B — 0005: Spam classification baseline — Logistic Regression

title: Spam classification baseline — Logistic Regression  
status: draft

scope (canonical files)
- openspec/ml/spam/README.md
- openspec/ml/spam/requirements.txt
- openspec/ml/spam/train_baseline.py
- openspec/ml/spam/predict.py
- openspec/ml/spam/tests/test_pipeline.py
- openspec/ml/spam/data_cache/sample_small.csv (cached sample for CI)
- openspec/web/streamlit_app.py (presentation/demo)

data
- Primary source:
  https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
- CI uses cached sample to avoid network dependency

design highlights
- Pipeline: download/validate → normalize → TfidfVectorizer(min_df=2, ngram_range=(1,2)) → stratified split → LogisticRegression (default) → save model + vectorizer (joblib) → write JSON evaluation report
- CLI: --data-url --out-dir --test-size --seed --model (logistic|svm)

Streamlit demo (integrated)
- Purpose: quick interactive demo for stream/meetings to inspect dataset, run inference, and showcase metrics.
- Main file: openspec/web/streamlit_app.py
- Minimal features:
  - Load cached sample or upload CSV
  - Display dataset preview and class distribution
  - Run inference using saved model artifacts and show metrics
  - Interactive text input for ad-hoc prediction
- How to run locally:
  - install `openspec/web/requirements.txt`
  - run: `streamlit run openspec/web/streamlit_app.py`

AI-assisted implementation steps
1. Add openspec change file (openspec/changes/0005-spam-classification.md).
2. Run: `ai-coding-cli scaffold --spec openspec/changes/0005-spam-classification.md`
   - Generates train_baseline.py, predict.py, unit tests, and a requirements file.
3. Manually review generated code and adjust preprocessing/thresholds.
4. Run: `ai-coding-cli verify --spec openspec/changes/0005-spam-classification.md` (lint + pytest + smoke train on cached sample)
5. Commit reviewed artifacts and open PR referencing the change file.

testing & CI
- Unit tests in openspec/ml/spam/tests and tests/python/
- CI job: install openspec/ml/spam/requirements.txt → run pytest (includes small smoke train on cached sample)
- Streamlit is excluded from CI runs; validate via local manual demo only

success criteria
- Baseline trains reproducibly with same seed; artifacts saved; tests pass in CI
- Streamlit demo runs locally and shows model predictions and metrics during presentation

---

## Combined rollout & conventions

1. Requirements files:
   - openspec/requirements-tools.txt (simulator deps)
   - openspec/ml/spam/requirements.txt (ML deps)
   - openspec/web/requirements.txt (streamlit + minimal deps)
2. Directory conventions:
   - Python tools under openspec/tools and openspec/ml
   - Streamlit demo under openspec/web
   - Tests under tests/python and tests/cpp
3. CI recommendations:
   - Main workflow: lint + unit tests (Python & C++).
   - Optional/integration workflow: starts docker-compose.mosquitto.yml for integration smoke tests.
   - ML smoke training uses cached sample to limit runtime in CI.
4. Documentation:
   - Update openspec/project.md with phase summaries.
   - Add README snippets in each feature directory explaining how to run locally and for presentations.

Notes:
- Follow AGENTS.md before authoring further proposals or implementing major changes.
- Keep dependencies optional where possible and include cached ML sample for CI reproducibility.