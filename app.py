import os
import sys
import textwrap
from pathlib import Path


import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from dotenv import load_dotenv
import chromadb
import google.generativeai as genai

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

load_dotenv()


DATA_DIR   = Path("data")
VECTOR_DIR = Path("vector_store")
for p in [DATA_DIR, VECTOR_DIR]:
    p.mkdir(exist_ok=True)

API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

db_path = DATA_DIR / "acme.db"
conn    = pysqlite3.connect(db_path)

schema_sql = """
DROP TABLE IF EXISTS sales;
CREATE TABLE sales(
    id INTEGER PRIMARY KEY,
    month TEXT,
    region TEXT,
    revenue REAL
);

DROP TABLE IF EXISTS tickets;
CREATE TABLE tickets(
    id INTEGER PRIMARY KEY,
    opened_at TEXT,
    closed_at TEXT,
    response_minutes INTEGER
);

DROP TABLE IF EXISTS customers;
CREATE TABLE customers(
    customer_id INTEGER PRIMARY KEY,
    region TEXT
);

DROP TABLE IF EXISTS subs;
CREATE TABLE subs(
    sub_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    month TEXT,
    status TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

DROP TABLE IF EXISTS employee_satisfaction;
CREATE TABLE employee_satisfaction(
    employee_id INTEGER PRIMARY KEY,
    month TEXT,
    rating INT
);
"""
conn.executescript(schema_sql)
conn.commit()

# Insert sample rows
sales_rows = [
    ("2025-01", "North America", 120000),
    ("2025-01", "EMEA",           95000),
    ("2025-02", "North America", 130000),
    ("2025-02", "EMEA",           99000),
]

ticket_rows = [
    ("2025-07-10 09:00", "2025-07-10 09:45", 45),
    ("2025-07-11 14:20", "2025-07-11 14:40", 20),
    ("2025-07-12 11:10", "2025-07-12 11:55", 45),
    ("2025-07-12 15:00", "2025-07-12 16:30", 90),
]

customer_rows = [
    (1, "North America"),
    (2, "EMEA"),
    (3, "North America"),
    (4, "EMEA"),
    (5, "North America"),
]

sub_rows = [
    ("2025-01", 1, "active"),
    ("2025-01", 2, "active"),
    ("2025-01", 3, "active"),
    ("2025-01", 4, "active"),
    ("2025-01", 5, "active"),

    ("2025-02", 1, "active"),
    ("2025-02", 2, "active"),
    ("2025-02", 3, "churned"),
    ("2025-02", 4, "churned"),
    ("2025-02", 5, "active"),

    ("2025-03", 1, "active"),
    ("2025-03", 2, "active"),
    ("2025-03", 5, "churned"),

]

satisfaction_rows = [
    ("2025-01", 70),
    ("2025-02", 90),
    ("2025-03", 60),
]

conn.executemany(
    "INSERT INTO sales(month, region, revenue) VALUES (?, ?, ?)", sales_rows
)
conn.executemany(
    "INSERT INTO tickets(opened_at, closed_at, response_minutes) VALUES (?, ?, ?)",
    ticket_rows
)
conn.executemany(
    "INSERT INTO customers(customer_id, region) VALUES (?, ?)",
    customer_rows
)
conn.executemany(
    "INSERT INTO subs(month, customer_id, status) VALUES (?, ?, ?)",
    sub_rows
)
conn.executemany(
    "INSERT INTO employee_satisfaction(month, rating) VALUES (?, ?)",
    satisfaction_rows
)

conn.commit()
print("Inserted rows:", 
      conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0], "sales –", 
      conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0], "tickets –", 
      conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0], "customers –",
      conn.execute("SELECT COUNT(*) FROM subs").fetchone()[0], "subscription updates –", 
      conn.execute("SELECT COUNT(*) FROM employee_satisfaction").fetchone()[0], "employee satisfaction ratings"
)



docs = {
    "security_policy": """
ACME enforces MFA and rotating secrets every 90 days.
Customer data is encrypted at rest and in transit.
""",
    "code_review": """
Every pull request needs two approvals. Linting and unit tests must pass in CI.
""",
    "onboarding": """
New hires receive a laptop, configure VPN access, and finish security training during week 1.
""",
    "customer complaints": """
Customer complaints are handled in a timely manner (typically within 2 hours), routed to the appropriate
department, and resolved with in the business day
""",
}

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))
collection = chroma_client.get_or_create_collection("acme_docs")

for key, text in docs.items():
    collection.add(
        documents=[text.strip()],
        ids=[key],
        metadatas=[{"source": "hand‑written"}]
    )

#print("Vector store size:", collection.count())

def qualitative_answer(query: str, k: int = 3) -> dict:
    results = collection.query(query_texts=[query], n_results=k)
    contexts = results["documents"][0]
    prompt = textwrap.dedent(f"""
        Answer the user question using only the provided context.
        Cite sources by enclosing the document ID in brackets.
        If the answer is unknown, say "I am not sure."

        Context:
        {chr(10).join(f"[{doc_id}] {ctx}" for doc_id, ctx in zip(results["ids"][0], contexts))}

        Question: {query}
        Answer:
    """)
    resp = model.generate_content(prompt, generation_config={"temperature": 0.9})
    return {"type": "qualitative", "answer": (resp.text or "").strip(), "sources": results["ids"][0]}

#print(qualitative_answer("What is the policy on secret rotation?")["answer"])

def safe_sql(query: str) -> dict:
    schema = """
    Table sales(id, month, region, revenue)
    Table tickets(id, opened_at, closed_at, response_minutes)
    Table customers(customer_id, region)
    TABLE subs(sub_id, customer_id, month, status)
    TABLE employee_satisfaction(employee_id, month, rating)
    """

    prompt = textwrap.dedent(f"""
        You are an SQL expert. Convert the user's question into a valid SQLite query
        that matches the given schema. Return only the SQL—no markdown fences.

        {schema}

        Question: {query}
        SQL:
    """)

    try: 
        resp = model.generate_content(prompt)
        raw_sql = (resp.text or "").strip()
    except Exception as e:
        return {"type": "quantitative", "error": f"There was an LLM Error: {e}", "sql": ""}
    

    # Clean possible ```sql fences just in case
    sql = (
        raw_sql.replace("```sql", "")
               .replace("```", "")
               .strip()
    )

    try:
        df = pd.read_sql_query(sql, conn)
        return {
            "type": "quantitative",
            "sql": sql,
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        return {"type": "quantitative", "error": str(e), "sql": sql}


#safe_sql("Show total revenue by region for 2025‑01")


def classify(q: str) -> str:
    numeric_cues = ["total", "sum", "average", "count", "rate", "trend", "revenue", "minutes", "percent", 
                    "%", "month", "monthly", "year", "yearly", "week", "weekly", "compare", "rate", "chart",
                    "per"]
    qual_cues = ["policy", "policies", "process", "procedure", "explain", "what is", "how do", "how to",
                 "review", "document", "docs"]
    
    text=q.lower()

    qual_hit = any(cue in text for cue in qual_cues)
    quant_hit = any(cue in text for cue in numeric_cues)
    
    if qual_hit and quant_hit:
        return "mixed"
    if  quant_hit:
        return "quantitative"
    return "qualitative"

def manager(q: str) -> dict:
    kind = classify(q)
    if kind == "qualitative":
        return qualitative_answer(q)
    if kind == "quantitative":
        return safe_sql(q)
    qa    = qualitative_answer(q)
    quant = safe_sql(q)
    return {"type": "mixed", "qualitative": qa, "quantitative": quant}


app = typer.Typer(help="ACME Multi‑Agent RAG Assistant")
console = Console()

@app.command()
def ask(
    query: str = typer.Argument(..., help = "Your question to the RAG System")
):
    """Ask a question about ACME docs or data."""
    result = manager(query)

    if result["type"] == "qualitative":
        console.print(result["answer"])
        if result.get("sources"):
            console.print(f"[dim]Sources: {', '.join(result['sources'])}[/dim]")
    elif result["type"] == "quantitative":
        if "error" in result:
            console.print(f"[red]SQL Error:[/red] {result['error']}")
            console.print(result["sql"])
        else:
            console.print(f"[dim]SQL: {result['sql']}[/dim]")
            rows = result.get("data") or []
            if rows: 
                headers = list(rows[0].keys())
                table = Table(show_header=True)
                for h in headers:
                    table.add_column(str(h))
                for row in rows:
                    table.add_row(*[str(row.get(h,"")) for h in headers])
                console.print(table)
            else:
                console.print("[yellow]No rows were returned[/yellow]")
    else: #Mixed

        #Qualitative
        console.rule("Qualitative")
        qa = result.get("qualitative", {})
        console.print(qa.get("answer"))
        if qa.get("sources"):
            console.print(f"[dim]Sources: {', '.join(qa['sources'])}[/dim]")
       
        #Quantitative
        console.rule("Quantitative")
        qres = result.get("quantitative", {})
        if qres.get("error"):
            console.print(f"[red]SQL Error:[/red] {qres['error']}")
            if qres.get("sql"):
                console.print(qres["sql"])
        else:
            sql_text = qres.get("sql", "")
            if(sql_text):
                console.print((f"[dim]SQL: {sql_text}[dim]"))
            rows = qres.get("data") or []
            if rows: 
                headers = list(rows[0].keys())
                table = Table(show_header=True)
                for h in headers:
                    table.add_column(str(h))
                for row in rows:
                    table.add_row(*[str(row.get(h,"")) for h in headers])
                console.print(table)
            else:
                console.print("[yellow]No rows were returned[/yellow]")

if __name__ == "__main__" and "ipykernel" not in sys.modules:
    app()