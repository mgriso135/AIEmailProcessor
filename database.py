import sqlite3

def create_connection():
    """ Create a database connection to the SQLite database """
    conn = None
    try:
        conn = sqlite3.connect('emails.db')
        if conn:
           create_table(conn)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn):
    """ Create a table if it doesn't exists"""
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT UNIQUE,
                contact_name TEXT,
                company TEXT,
                email_address TEXT,
                phone_number TEXT,
                category TEXT,
                priority TEXT,
                email_body TEXT,
                email_date TEXT,
                processed INTEGER DEFAULT 0,
                language TEXT
            )
        """)
        conn.commit()
        print("Table 'emails' created or exists already.")
    except sqlite3.Error as e:
         print(f"Error creating table: {e}")

def insert_email(conn, email_data):
    """ Insert email data into the emails table """
    sql = ''' INSERT INTO emails(email_id, contact_name, company, email_address, phone_number, category, priority, email_body, email_date, processed, language)
              VALUES(?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, email_data)
    conn.commit()
    return cur.lastrowid

def update_email_processed(conn, email_id):
    """Updates email as processed"""
    sql = ''' UPDATE emails SET processed = 1 WHERE email_id = ?'''
    cur = conn.cursor()
    cur.execute(sql, (email_id,))
    conn.commit()

def get_emails(conn, unprocessed_only=True):
    """ Get emails from the database """
    cur = conn.cursor()
    if unprocessed_only:
        cur.execute("SELECT * FROM emails WHERE processed = 0")
    else:
        cur.execute("SELECT * FROM emails")
    rows = cur.fetchall()
    return rows
def email_exists(conn, email_id):
    """ Check if an email exists in the database """
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM emails WHERE email_id = ?", (email_id,))
    return cur.fetchone() is not None

if __name__ == '__main__':
    conn = create_connection()
    if conn:
        create_table(conn)
        conn.close()