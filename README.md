# Support Pilot - AI Ticket Resolution agent (Milestone 1). 
## Overview
Support Pilot is an AI-Powered Ticket resolution agent that is created with python. It allows you to manage tickets that help you to solve issues, predict severity, and assign priorities to the tickets, depending on the severity of the issue.

## Features

- Interactive Dashboard
- Full CLI Parity
- Rule-based severity and priority triage

## Prerequisites
- **Python 3**
- **Flask**
- **Pandas**
- **SciKit Learn**

## Project Structure
```text
SupportPilot_AI/
├── app.py
├── cli.py
├── classifier.py
├── database.py
├── templates/ 
├── ├── index.html
├── train_model.py
└── requirements.txt 

```

## How to install
- Clone this repository.
```bash
git clone https://github.com/ashw445/SupportPilot_AI/
cd SupportPilot_AI
```
- Install Prerequisites
```bash
pip install -r requirements.txt
```
or
```bash
pip install pandas flask scikitlearn
```

## How to run
- Train the model.
```bash
python train_model.py
```
- Run the app.
```bash
python app.py
```
- Run the CLI (optional).
```
- python train_model.py
```
## Creator
Ashwin Raghu, for Infosys Springboard Internship
