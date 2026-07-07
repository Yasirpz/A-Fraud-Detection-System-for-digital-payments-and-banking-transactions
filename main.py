#!/usr/bin/env python3
"""
main.py
-------
Single entry point for the whole Fraud Detection AI/ML pipeline
(System G: Fraud Detection System for Digital Payments and Banking
Transactions -- AI Course, PEAS Assignment 1 + Lab 1 practical module).

Subcommands:
    generate   Build the synthetic transaction dataset
    train      Train and evaluate the RandomForest fraud model
    predict    Score transactions via CLI (demo / interactive / json / flags)

Quick start (from repo root):
    pip install -r requirements.txt
    python main.py generate
    python main.py train
    python main.py predict --demo
"""

import argparse
import runpy
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).parent / "data"))


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Fraud Detection AI System -- unified CLI entry point.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="Generate the synthetic transactions dataset")
    subparsers.add_parser("train", help="Train and evaluate the fraud detection model")
    subparsers.add_parser("predict", help="Score a transaction (see: python main.py predict --help)")

    args, remaining = parser.parse_known_args()

    if args.command == "generate":
        sys.argv = ["generate_dataset.py"] + remaining
        runpy.run_path(str(Path(__file__).parent / "data" / "generate_dataset.py"), run_name="__main__")
    elif args.command == "train":
        sys.argv = ["train.py"] + remaining
        runpy.run_path(str(SRC_DIR / "train.py"), run_name="__main__")
    elif args.command == "predict":
        sys.argv = ["predict.py"] + remaining
        runpy.run_path(str(SRC_DIR / "predict.py"), run_name="__main__")


if __name__ == "__main__":
    main()
