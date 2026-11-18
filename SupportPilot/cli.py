import os
import re
import sys
import joblib
import pandas as pd
from flask import Flask
from database import initialize_database, insert_ticket, get_all_tickets
from classifier import classify_text

MODEL_PATH = os.path.join("models", "ticket_classifier.pkl")
DB_PATH = "tickets.db"
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"File {MODEL_PATH} not found. Run 'python train_model.py' first.")
        sys.exit(1)


def input_not_empty(prompt: str, field: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print(f"{field} cannot be empty. Please try again.")


def input_valid_email(prompt: str) -> str:
    while True:
        email = input_not_empty(prompt, "Email")
        if re.match(EMAIL_PATTERN, email):
            return email
        print("Invalid email format. Please try again.")


def process_and_save_ticket(model, employee_name, email, department, title, description):
    full_text = f"{title}: {description}"
    prediction = classify_text(full_text, model)
    if not prediction:
        print(f"Ticket '{title}' could not be classified.")
        return None

    return insert_ticket(
        employee_name=employee_name,
        email=email,
        department=department,
        title=title,
        description=description,
        category=prediction["category"],
        severity=prediction["severity"],
        priority=prediction["priority"],
        confidence=prediction["confidence"],
    )


def submit_ticket(model):
    employee_name = input_not_empty("Enter the employee name: ", "Employee Name")
    email = input_valid_email("Enter the email address of the employee: ")
    department = input_not_empty("Enter the department of the employee: ", "Department")
    title = input_not_empty("Enter the title of the issue: ", "Title")
    description = input_not_empty("Enter the description of the issue: ", "Description")

    ticket_id = process_and_save_ticket(
        model, employee_name, email, department, title, description
    )
    if ticket_id:
        print(f"Ticket created successfully! ID: {ticket_id}")


def submit_tickets_from_csv(model, csv_file="employee_data.csv"):
    try:
        df = pd.read_csv(csv_file).fillna("")
    except FileNotFoundError:
        print(f"Error: '{csv_file}' not found.")
        return

    required_cols = {"Employee Name", "Email", "Department", "Title", "Description"}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV is missing required columns: {required_cols - set(df.columns)}")
        return

    for idx, row in df.iterrows():
        name = str(row["Employee Name"]).strip()
        email = str(row["Email"]).strip()
        dept = str(row["Department"]).strip()
        title = str(row["Title"]).strip()
        desc = str(row["Description"]).strip()

        if not all([name, email, dept, title, desc]) or not re.match(EMAIL_PATTERN, email):
            print(f"Row {idx + 1} skipped: missing fields or invalid email ({email}).")
            continue

        ticket_id = process_and_save_ticket(model, name, email, dept, title, desc)
        if ticket_id:
            print(f"Row {idx + 1}: Ticket created successfully! ID: {ticket_id}")


def view_all_tickets():
    if not os.path.exists(DB_PATH):
        print(f"{DB_PATH} not found.")
        return

    tickets = get_all_tickets(DB_PATH)
    if not tickets:
        print("No tickets found.")
        return

    print("\n" + "-" * 50)
    for t in tickets:
        print(
            f"[{t['ticket_id']}] {t['title']} | {t['category']} | "
            f"{t['severity']} ({t['priority']}) | Status: {t['status']} | "
            f"By: {t['employee_name']} ({t['department']}) at {t['created_at']}"
        )
    print("-" * 50 + "\n")


def print_menu():
    print("\n" + "=" * 50)
    print("Support Pilot - AI-powered Ticket Classifier")
    print("=" * 50)
    print("1. Submit a ticket")
    print("2. Submit tickets from CSV")
    print("3. View all tickets")
    print("4. Quit")
    print("=" * 50)


def main():
    initialize_database(DB_PATH)
    model = load_model()

    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()
        if choice == "1":
            submit_ticket(model)
        elif choice == "2":
            submit_tickets_from_csv(model)
        elif choice == "3":
            view_all_tickets()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()