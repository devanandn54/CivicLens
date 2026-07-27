import psycopg, os
from dotenv import load_dotenv
load_dotenv()
with psycopg.connect(os.environ['DATABASE_URL']) as conn, conn.cursor() as cur:
    cur.execute('SELECT id, title, source_url FROM meetings ORDER BY id')
    for row in cur.fetchall():
        print(row)