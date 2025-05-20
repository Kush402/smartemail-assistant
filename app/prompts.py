# Prompt def build_prompt(subject: str, email_body: str) -> str:
    return f"""Write a professional email response to:
Subject: {subject}

Original Email:
{email_body}
"""
templates and generation logic 