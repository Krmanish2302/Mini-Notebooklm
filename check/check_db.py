import sqlite3

def check_db():
    import os
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test', 'test_reg.db'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, source_type, title FROM sources")
    sources = cursor.fetchall()
    print("Sources:", sources)
    
    for src in sources:
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE source_id=?", (src[0],))
        count = cursor.fetchone()[0]
        print(f"Chunks for {src[2]} ({src[1]}): {count}")
        
        if count > 0:
            cursor.execute("SELECT content FROM chunks WHERE source_id=? LIMIT 1", (src[0],))
            sample = cursor.fetchone()[0]
            print(f"Sample chunk: {sample[:200]}...\n")

if __name__ == "__main__":
    check_db()
