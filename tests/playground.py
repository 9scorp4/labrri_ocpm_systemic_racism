import sqlite3
import re

# Connect to the database
conn = sqlite3.connect('data/database.db')
cursor = conn.cursor()

# Define the minimum content length threshold
min_content_length = 500
min_concatenated_length = 100

# Filter step 1: Checking documents with concatenated words without spaces
cursor.execute("""
    SELECT filepath, content
    FROM documents
    INNER JOIN content ON documents.id = content.doc_id
""")
rows = cursor.fetchall()

concatenated_documents = []
for filepath, content in rows:
    # Find sequences of uppercase and lowercase letters
    sequences = re.findall(r'[A-ZÀ-ȕ][a-zà-ȕ]+', content)
    # Filter sequences longer than the threshold
    lengthy_sequences = [seq for seq in sequences if len(seq) >= min_concatenated_length]
    if lengthy_sequences:
        concatenated_documents.append(filepath)

# Filter step 2: Checking documents with unreadable content
cursor.execute("""
            SELECT filepath
            FROM documents
            INNER JOIN content ON documents.id = content.doc_id
            WHERE LENGTH(TRIM(content.content)) < ?
        """, (min_content_length,))
rows = cursor.fetchall()

unreadable_documents = []
for filepath in rows:
    filepath = filepath[0]
    unreadable_documents.append(filepath)

# Show results
results = concatenated_documents + unreadable_documents
print(results)

# Close the database connection
conn.close()
