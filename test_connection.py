"""
Day 0 verification gate.
Fill in your AACT username/password below, then run this file.
If it prints a study count (500,000+) and a status breakdown, you are ready for Day 1.
"""
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import pandas as pd
# ---- FILL THESE IN ----
USER = "dc13"
PW = "your_aact_password"
# -----------------------
engine = create_engine(
    f"postgresql+psycopg2://{quote_plus(USER)}:{quote_plus(PW)}@aact-db.ctti-clinicaltrials.org:5432/aact"
)
# Test 1: can we connect and count studies?
n = pd.read_sql("SELECT COUNT(*) AS n FROM studies;", engine)
print("Total studies in AACT:", n["n"][0])
# Test 2: peek at the status field we'll label on in Day 1
print("\nStatus breakdown:")
print(pd.read_sql("""
    SELECT overall_status, COUNT(*) AS n
    FROM studies
    GROUP BY overall_status
    ORDER BY n DESC
    LIMIT 10;
""", engine))