import re

def clean_email(text: str) -> str:
    # Basic cleaning: remove extra spaces and line breaks
    return re.sub(r'\s+', ' ', text).strip()

def format_prompt(subject: str, body: str) -> str:
    return f"Write a professional email response to:\nSubject: {subject}\n\nOriginal Email:\n{body}"
