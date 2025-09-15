This project uses a Multi-Agent RAG Assistant that can answer two forms of general inquiries: 

-- Qualitative Queries: 
  -- Using documentation and semantic search via ChromaDB
-- Quantitative Queries: 
  -- Using SQL queries via an SQLite Database

The Assistant can also answer queries that combine aspects of qualitative and quantitative queries. 


The assistant can answer queries based on the given data, which for for a company 'ACME', with data including security policy, onboarding processes, sales data, employee satisfaction, and more



Answer queries in the CL by running in the following format: 'python app.py "What is our customer churn rate by month?"'

Sample queries include: 
  Qualitative Queries (Semantic/Document):

    "What is our company's security policy?"

    "Explain the code review process"

    "How do we handle customer complaints?"

Quantitative Queries (SQL/Numerical):

    "Show me monthly revenue trends"

    "What's our customer churn rate?"

    "Compare Q4 performance across regions"

