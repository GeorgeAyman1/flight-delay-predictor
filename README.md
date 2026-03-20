# Flight Delay Prediction

## Overview
This project predicts whether a flight will be delayed using machine learning.

## Business Problem
Flight delays create operational, financial, and customer-service problems for airlines and airports.  
The goal is to predict delays early enough to support better decisions.

## Stakeholder
- Airline operations teams
- Airport operations teams
- Travel platforms / passengers

## Project Goal
Build a supervised classification model that predicts whether a flight will be delayed.

## Team
- Jumana Timor
- Youssef El-sawy
- Lujina Shawki
- George Ayman

## Planned Data Sources
- Flight performance data
- Weather data
- Airport / route metadata

## Repository Structure
flight-delay-predictor/
├── .github/                 # GitHub workflows
├── dashboards/              # Dashboards / presentation outputs
├── data/
│   ├── raw/                 # Raw input data (kept local, ignored by Git)
│   ├── interim/             # Intermediate data (kept local, ignored by Git)
│   └── processed/           # Final processed data (kept local, ignored by Git)
├── notebooks/               # Jupyter notebooks for exploration
├── reports/
│   ├── figures/             # Charts and plots
│   └── tables/              # Generated tables
├── src/
│   ├── data/                # Data acquisition, validation, and merging
│   ├── features/            # Feature engineering
│   ├── models/              # Model training / prediction
│   ├── evaluation/          # Evaluation and metrics
│   └── utils/               # Utility/helper functions
├── tests/                   # Tests
├── .env.example             # Example environment variables
├── .gitignore               # Ignore rules
├── Makefile                 # Shortcut commands
├── pyproject.toml           # Project configuration
├── poetry.lock              # Dependency lock file
└── README.md                # Project overview and instructions

## Git Workflow
- `main` = stable branch
- each member works on a feature branch
- changes are merged through pull requests

## Status
Project setup in progress.

## This project belongs to Data Science course, in Cairo University Faculty of Engnineering
