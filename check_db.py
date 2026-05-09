import sqlite3

def check_db():
    conn = sqlite3.connect("test_reg.db")
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
