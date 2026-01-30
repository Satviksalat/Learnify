
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR PYTHON UNIT 5
PYTHON_UNIT_5_DATA = {
    "unit": "Python Unit 5: Database Connectivity",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "Define Data Connectivity.", "answer": "The ability of a programming language to connect and interact with a database system."},
            {"question": "What is Python DB-API?", "answer": "The standard API (Application Programming Interface) for accessing databases in Python."},
            {"question": "What is MySQL?", "answer": "An open-source Relational Database Management System (RDBMS) that uses SQL."},
            {"question": "What is a 'connector' in Python?", "answer": "A module or driver that acts as a bridge between Python and a specific database (e.g., mysql-connector)."},
            {"question": "What does 'connect()' do?", "answer": "A function that establishes a connection to the database and returns a connection object."},
            {"question": "What is a Cursor object?", "answer": "An object used to execute SQL queries and fetch data from the database."},
            {"question": "What is 'execute()'? method", "answer": "A cursor method used to run a single SQL query."},
            {"question": "What is 'commit()'? method", "answer": "A connection method used to save changes (INSERT, UPDATE, DELETE) permanently to the database."},
            {"question": "What is 'rollback()'? method", "answer": "A method to revert changes made during the current transaction if an error occurs."},
            {"question": "What is 'fetchall()'?", "answer": "A method that retrieves all rows of a query result as a list of tuples."},
            {"question": "What is 'fetchone()'?", "answer": "A method that retrieves the next single row of a query result."},
            {"question": "Define SQL Injection.", "answer": "A security vulnerability where an attacker interferes with the queries an application makes to its database."},
            {"question": "What is a Primary Key?", "answer": "A unique identifier for each record in a database table."},
            {"question": "What is 'close()' method?", "answer": "Used to close the cursor and connection to free up resources."},
            {"question": "Define DDL.", "answer": "Data Definition Language: Commands that define the structure (CREATE, DROP, ALTER)."},
            {"question": "Define DML.", "answer": "Data Manipulation Language: Commands that manipulate data (INSERT, UPDATE, DELETE)."},
            {"question": "What is 'rowcount'?", "answer": "A property of the cursor that returns the number of rows affected by the last executed query."},
            {"question": "What is parameterized query?", "answer": "Using placeholders (%s) in SQL queries to prevent SQL injection."},
            {"question": "What is a Result Set?", "answer": "The set of rows returned by a SELECT query."},
            {"question": "Default port for MySQL?", "answer": "3306."}
        ],
        "Part B (2-Marks)": [
            {"question": "Steps to connect Python to MySQL.", "answer": "• Import connector.\n• Establish connection (host, user, pass).\n• Create cursor.\n• Execute queries."},
            {"question": "fetchall() vs fetchmany(n).", "answer": "• fetchall(): Gets ALL rows at once (memory intensive).\n• fetchmany(n): Gets 'n' rows at a time."},
            {"question": "Commit vs Rollback.", "answer": "• Commit: Saves changes. Transaction successful.\n• Rollback: Undoes changes. Transaction failed."},
            {"question": "Cursor vs Connection Object.", "answer": "• Connection: Handles authentication and transaction state.\n• Cursor: Handles query execution and result traversal."},
            {"question": "Why close connection?", "answer": "• To release memory and network resources.\n• To avoid 'Too many connections' error on the server."},
            {"question": "Explain Insert Query syntax in Python.", "answer": "• sql = 'INSERT INTO table (col) VALUES (%s)'\n• val = ('value',)\n• cursor.execute(sql, val)"},
            {"question": "What is 'mysql-connector-python'?", "answer": "• A standardized database driver provided by Oracle.\n• Allows Python programs to access MySQL databases."},
            {"question": "Handling Database Errors.", "answer": "• Use try-except blocks.\n• Catch 'mysql.connector.Error' to display meaningful messages."},
            {"question": "DDL vs DML.", "answer": "• DDL: CREATE table, DROP table (Structure).\n• DML: INSERT, UPDATE, DELETE (Data)."},
            {"question": "Update Query Logic.", "answer": "• Change existing data based on a condition.\n• 'UPDATE table SET col=val WHERE id=1'."},
            {"question": "Delete vs Drop.", "answer": "• Delete: Removes rows from table (Table structure remains).\n• Drop: Removes the entire table from database."},
            {"question": "What is WHERE clause?", "answer": "• Used to filter records.\n• Essential for UPDATE/DELETE to prevent affecting all rows."},
            {"question": "Placeholders (%s) usage.", "answer": "• Used for binding variables to queries.\n• Python automatically escapes inputs to prevent security risks."},
            {"question": "Check if connection is established.", "answer": "• Use `if connection.is_connected():`\n• Returns True if connection is active."},
            {"question": "Creating a Table in Python.", "answer": "• Write 'CREATE TABLE' SQL string.\n• Call `cursor.execute(sql_string)`."}
        ],
        "Part C (3-Marks)": [
            {"question": "Connecting to Database (Code).", "answer": "1. **Definition:** Setup.\n2. **Explanation:** `con = connector.connect(host='localhost', user='root', password='')`. Check `if con.is_connected(): print('Success')`.\n3. **Example:** Standard boilerplate code."},
            {"question": "Creating a Table (Code).", "answer": "1. **Definition:** Structure definition.\n2. **Explanation:** SQL = `CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255))`. Execute it.\n3. **Example:** Creating a Student register."},
            {"question": "Insert Data (Code).", "answer": "1. **Definition:** Adding records.\n2. **Explanation:** SQL = `INSERT INTO users (name) VALUES (%s)`. Val = `('John',)`. `cursor.execute(sql, val)`. `con.commit()`.\n3. **Example:** Adding a new user sign-up."},
            {"question": "Read/Select Data (Code).", "answer": "1. **Definition:** Fetching records.\n2. **Explanation:** `cursor.execute('SELECT * FROM users')`. `rows = cursor.fetchall()`. Loop `for r in rows: print(r)`.\n3. **Example:** Displaying a list of products."},
            {"question": "Update Data (Code).", "answer": "1. **Definition:** Modifying records.\n2. **Explanation:** SQL = `UPDATE users SET age=20 WHERE name='John'`. `cursor.execute(sql)`. `con.commit()`.\n3. **Example:** Changing a password."},
            {"question": "Delete Data (Code).", "answer": "1. **Definition:** Removing records.\n2. **Explanation:** SQL = `DELETE FROM users WHERE id=1`. `cursor.execute(sql)`. `con.commit()`. Check `cursor.rowcount`.\n3. **Example:** Deleting a cancelled order."},
            {"question": "Preventing SQL Injection.", "answer": "1. **Definition:** Security practice.\n2. **Explanation:** NEVER use string formatting (f-strings) for queries. ALWAYS use parameterized queries with `%s` tuple.\n3. **Example:** Bad: `...VALUES ('\" + user + \"')`. Good: `...VALUES (%s), (user,)`."},
            {"question": "Transaction Management.", "answer": "1. **Definition:** Atomicity.\n2. **Explanation:** Ensure all operations succeed or none do. Start -> Try Ops -> Commit. Except -> Rollback.\n3. **Example:** Bank transfer (Debit A, Credit B)."},
            {"question": "Executing Multiple Inserts.", "answer": "1. **Definition:** Bulk loading.\n2. **Explanation:** Use `cursor.executemany(sql, list_of_tuples)`. Faster than loop.\n3. **Example:** Importing 100 students from a CSV."},
            {"question": "Handling Date/Time in SQL.", "answer": "1. **Definition:** Temporal data.\n2. **Explanation:** Python `datetime` objects map to SQL `DATETIME`. Pass datetime object as parameter.\n3. **Example:** `INSERT INTO logs (time) VALUES (%s), (datetime.now(),)`."},
            {"question": "Order By Clause.", "answer": "1. **Definition:** Sorting results.\n2. **Explanation:** `SELECT * FROM table ORDER BY col ASC|DESC`.\n3. **Example:** Showing high scores (DESC order)."},
            {"question": "Limit Clause.", "answer": "1. **Definition:** Pagination.\n2. **Explanation:** `SELECT * FROM table LIMIT 5 OFFSET 10`. Get 5 records starting from 11th.\n3. **Example:** Page 2 of google results."},
            {"question": "Why use 'with' statement?", "answer": "1. **Definition:** Context Manager.\n2. **Explanation:** Ensures connection closes automatically. `with connector.connect(...) as con:`.\n3. **Example:** Best practice for resource management."},
            {"question": "Primary vs Foreign Key.", "answer": "1. **Definition:** Relational integrity.\n2. **Explanation:** Primary: Unique ID for row. Foreign: Reference to Primary key of another table.\n3. **Example:** StudentID (Primary) in Students, StudentID (Foreign) in Grades."},
            {"question": "Fetching Query Metadata.", "answer": "1. **Definition:** Column info.\n2. **Explanation:** cursor.description contains column names and types after a SELECT.\n3. **Example:** Printing headers for a CSV export."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Python DB-API.",
                "answer": "1. **Definition:**\n   The Python Database API Specification (PEP 249) that defines a common interface for accessing databases.\n\n2. **Goal:**\n   To ensure consistency. Code written for MySQL should work for SQLite/Postgres with minimal changes.\n\n3. **Core Concept:**\n   Standard Objects: Connection, Cursor. Standard Methods: connect, execute, fetch, commit, rollback.\n\n4. **Technique / Method:**\n   Any DB driver (psycopg2, mysql-connector) follows this standard structure.\n\n5. **Applications:**\n   Allows switching database backends easily in frameworks like Django or SQLAlchemy."
            },
            {
                "question": "Detailed Steps for Database Connectivity.",
                "answer": "1. **Definition:**\n   The complete workflow to interact with a database from a Python script.\n\n2. **Goal:**\n   To establish a secure channel to read/write persistent data.\n\n3. **Core Concept:**\n   1. Import. 2. Connect. 3. Cursor. 4. Execute. 5. Commit (if needed). 6. Close.\n\n4. **Technique / Method:**\n   `db = connect(...)`. `cur = db.cursor()`. `cur.execute(...)`. `db.commit()`. `db.close()`.\n\n5. **Applications:**\n   Backend APIS, Data Migration scripts, Desktop apps with local storage."
            },
            {
                "question": "Detailed Explanation of CRUD Operations.",
                "answer": "1. **Definition:**\n   The four basic functions of persistent storage: Create, Read, Update, Delete.\n\n2. **Goal:**\n   To maintain complete control over the data lifecycle.\n\n3. **Core Concept:**\n   Create=INSERT. Read=SELECT. Update=UPDATE. Delete=DELETE.\n\n4. **Technique / Method:**\n   Each corresponds to a DML SQL command executed via cursor.execute().\n\n5. **Applications:**\n   Every dynamic website (Post a blog, Read it, Edit it, Delete it)."
            },
            {
                "question": "Detailed Explanation of SQL Injection & Prevention.",
                "answer": "1. **Definition:**\n   A code injection technique where an attacker destroys your database or steals data.\n\n2. **Goal:**\n   Security. To ensure user input is treated as data, not executable code.\n\n3. **Core Concept:**\n   Never trust user input. Separation of Code (SQL) and Data (Parameters).\n\n4. **Technique / Method:**\n   Vulnerable: `sql = 'SELECT * FROM u WHERE name=' + user`. Secure: `sql = '... name=%s'`. Driver handles escaping.\n\n5. **Applications:**\n   Login forms (preventing ' OR 1=1 -- ' attacks)."
            },
            {
                "question": "Detailed Explanation of Transactions (Commit/Rollback).",
                "answer": "1. **Definition:**\n   A logical unit of processing in a DBMS involving one or more database access operations.\n\n2. **Goal:**\n   Data Integrity. Validates that the database remains in a consistent state.\n\n3. **Core Concept:**\n   ACID properties (Atomicity). All or Nothing.\n\n4. **Technique / Method:**\n   Default behavior is Auto-Commit off in most python drivers. You MUST call `.commit()` to save. Call `.rollback()` on error.\n\n5. **Applications:**\n   Transferring money involved debiting one row and crediting another. Both must succeed."
            },
            {
                "question": "Detailed Explanation of Cursors.",
                "answer": "1. **Definition:**\n   A control structure that enables traversal over the records in a database.\n\n2. **Goal:**\n   To process individual rows returned by a query or execute commands.\n\n3. **Core Concept:**\n   Pointer. It points to the current context/row.\n\n4. **Technique / Method:**\n   `cursor = conn.cursor()`. Buffered cursors fetch all data to client. Unbuffered fetch on demand.\n\n5. **Applications:**\n   Iterating through 1 million user records one by one without loading all into RAM."
            },
            {
                "question": "Detailed Explanation of Data Retrieval Methods.",
                "answer": "1. **Definition:**\n   Methods used to extract data from a Result Set after a SELECT query.\n\n2. **Goal:**\n   To bring database data into Python variables for processing.\n\n3. **Core Concept:**\n   Fetching. Pulling data from the server.\n\n4. **Technique / Method:**\n   `fetchone()`: Next row (Tuple). `fetchall()`: List of all rows (List of Tuples). `fetchmany(size)`: Batch.\n\n5. **Applications:**\n   Displaying search results (10 per page), Exporting entire DB (fetchall)."
            },
            {
                "question": "Detailed Explanation of MySQL Connector.",
                "answer": "1. **Definition:**\n   A self-contained Python driver for communicating with MySQL servers.\n\n2. **Goal:**\n   To provide Pythonic implementation of the MySQL protocol.\n\n3. **Core Concept:**\n   Native Python. No C dependencies needed (unlike some other drivers).\n\n4. **Technique / Method:**\n   Installation: `pip install mysql-connector-python`. Usage: `import mysql.connector`.\n\n5. **Applications:**\n   Standard choice for Python-MySQL projects."
            },
            {
                "question": "Detailed Explanation of Insert with Dynamic Data.",
                "answer": "1. **Definition:**\n   Adding records where values come from variables/user input.\n\n2. **Goal:**\n   To save runtime data.\n\n3. **Core Concept:**\n   Parameterization. Binding variables to the query string safely.\n\n4. **Technique / Method:**\n   Define template with `%s`. Create tuple of values. Pass both to execute.\n\n5. **Applications:**\n   User Registration forms, IoT sensor data logging."
            },
            {
                "question": "Detailed Explanation of Deletion Logic.",
                "answer": "1. **Definition:**\n   Removing specific records from a table permanently.\n\n2. **Goal:**\n   To clean up obsolete or requested data.\n\n3. **Core Concept:**\n   Correct usage of WHERE clause. Missing WHERE deletes ALL rows.\n\n4. **Technique / Method:**\n   `sql = DELETE FROM t WHERE id=%s`. `val = (5,)`. Confirm with `commit()`. Check `rowcount` to see if it worked.\n\n5. **Applications:**\n   Account deletion, Removing temporary data, Archiving."
            }
        ]
    }
}

def populate_python_unit5():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Python Unit 5: Database Connectivity"
        data = [u for u in data if "Python Unit 5: Database Connectivity" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(PYTHON_UNIT_5_DATA)
        print("Successfully replaced Python Unit 5 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_python_unit5()
