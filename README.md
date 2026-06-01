---
title: MediScan
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# MediScan 🏥

![CI/CD](https://github.com/sunilakiran/mediscan/actions/workflows/ci-cd.yml/badge.svg)
![Lint](https://github.com/sunilakiran/mediscan/actions/workflows/lint.yml/badge.svg)

> AI-powered chest X-ray pneumonia detection with full MLOps pipeline.

## 🔗 Live Demo
[MediScan API on Hugging Face](https://huggingface.co/spaces/sunilakiran56/mediscan)

## 📦 Package
[mediscan-62671 on TestPyPI](https://test.pypi.org/project/mediscan-62671/)

## 🚀 Quick Start
docker compose up

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | API health check |
| /predict | POST | Upload X-ray, get diagnosis |
| /history | GET | Last 20 predictions |
| /stats | GET | Aggregated statistics |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| AUROC | 0.9453 |
| Recall | 0.9795 |
| F1 Score | 0.8957 |
| Precision | 0.8251 |