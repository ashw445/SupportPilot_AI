import io
import os
import re
import sys
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from database import (
    initialize_database,
    insert_ticket,
    get_all_tickets,
    update_ticket_status,
    delete_ticket,
    get_ticket_stats,
    VALID_STATUSES,
)
from classifier import classify_text

app = Flask(__name__)
app.secret_key = "support_pilot_secret_key"

MODEL_PATH = os.path.join("models", "ticket_classifier.pkl")
DB_PATH = "tickets.db"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: '{MODEL_PATH}' not found. Run 'python train_model.py' first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


initialize_database(DB_PATH)
model = load_model()


def process_and_save_ticket(employee_name, email, department, title, description):
    full_text = f"{title}: {description}"
    prediction = classify_text(full_text, model)
    if not prediction:
        return None

    try:
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
    except Exception as e:
        print(f"Database error: {e}")
        return None


@app.route("/", methods=["GET"])
def index():
    status_filter = request.args.get("status", "All")
    priority_filter = request.args.get("priority", "All")
    search_query = request.args.get("q", "").strip()

    tickets = get_all_tickets(
        DB_PATH,
        status_filter=status_filter,
        priority_filter=priority_filter,
        search_query=search_query,
    )
    stats = get_ticket_stats(DB_PATH)

    return render_template(
        "index.html",
        tickets=tickets,
        stats=stats,
        status_filter=status_filter,
        priority_filter=priority_filter,
        search_query=search_query,
        valid_statuses=sorted(VALID_STATUSES),
    )


@app.route("/submit", methods=["POST"])
def submit():
    employee_name = request.form.get("employee_name", "").strip()
    email = request.form.get("email", "").strip()
    department = request.form.get("department", "").strip()
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()

    if not all([employee_name, email, department, title, description]):
        flash("All fields are required.", "error")
        return redirect(url_for("index"))

    if not EMAIL_PATTERN.match(email):
        flash("Invalid email address format.", "error")
        return redirect(url_for("index"))

    ticket_id = process_and_save_ticket(employee_name, email, department, title, description)
    if ticket_id:
        flash(f"Ticket created successfully! (ID: {ticket_id})", "success")
    else:
        flash("Ticket could not be classified or saved.", "error")

    return redirect(url_for("index"))


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("Please select a CSV file to upload.", "error")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(file, dtype=str).fillna("")
    except Exception as e:
        flash(f"Error reading CSV file: {e}", "error")
        return redirect(url_for("index"))

    required_cols = {"Employee Name", "Email", "Department", "Title", "Description"}
    missing = required_cols - set(df.columns)
    if missing:
        flash(f"CSV is missing required columns: {', '.join(sorted(missing))}", "error")
        return redirect(url_for("index"))

    created_count = 0
    skipped_count = 0

    for _, row in df.iterrows():
        name = row["Employee Name"].strip()
        email = row["Email"].strip()
        dept = row["Department"].strip()
        title = row["Title"].strip()
        desc = row["Description"].strip()

        if not all([name, email, dept, title, desc]) or not EMAIL_PATTERN.match(email):
            skipped_count += 1
            continue

        if process_and_save_ticket(name, email, dept, title, desc):
            created_count += 1
        else:
            skipped_count += 1

    flash(f"CSV processed: {created_count} tickets created, {skipped_count} skipped.", "success")
    return redirect(url_for("index"))


@app.route("/update_status/<int:ticket_id>", methods=["POST"])
def change_status(ticket_id):
    new_status = request.form.get("status", "").strip()
    try:
        if update_ticket_status(ticket_id, new_status, DB_PATH):
            flash(f"Ticket #{ticket_id} updated to '{new_status}'.", "success")
        else:
            flash(f"Ticket #{ticket_id} not found.", "error")
    except ValueError as e:
        flash(str(e), "error")
    return redirect(url_for("index"))


@app.route("/delete_ticket/<int:ticket_id>", methods=["POST"])
def remove_ticket(ticket_id):
    if delete_ticket(ticket_id, DB_PATH):
        flash(f"Ticket #{ticket_id} deleted.", "success")
    else:
        flash(f"Ticket #{ticket_id} not found.", "error")
    return redirect(url_for("index"))


@app.route("/export_csv", methods=["GET"])
def export_csv():
    status_filter = request.args.get("status", "All")
    priority_filter = request.args.get("priority", "All")
    search_query = request.args.get("q", "").strip()

    tickets = get_all_tickets(
        DB_PATH,
        status_filter=status_filter,
        priority_filter=priority_filter,
        search_query=search_query,
    )

    df = pd.DataFrame(tickets)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=support_tickets_export.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True)