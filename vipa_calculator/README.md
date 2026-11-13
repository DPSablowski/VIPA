# VIPA Finesse Calculator

Calculator for the finesse of a real virtually imaged phased array (VIPA), with a graphical interface.

## Requirements

- Python 3 (3.9 or newer) with tkinter (included in the standard Windows and macOS installers; on Debian/Ubuntu: `sudo apt install python3-tk`)
- numpy, scipy, matplotlib, pandas

## Installation

    git clone <repository-url>
    cd <repository-folder>
    python -m pip install -r requirements.txt

Optionally in a virtual environment:

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    python -m pip install -r requirements.txt

## Start

    python start_calculator.py          # graphical interface
    python start_calculator.py --test   # self-test without GUI (about half a minute)

The self-test should end with "All checks passed".
