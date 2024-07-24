# -*- coding: utf-8 -*-

import sqlite3
import logging
import pandas as pd

class Database:
    """
    Class for database operations.
    """

    def __init__(self, db):
        """
        Initialize the database connection and cursor.

        Args:
            db (str): Path to the database.

        Raises:
            sqlite3.Error: If the database connection fails.
        """
        if not db:
            raise ValueError("db_path must be a non-empty string")

        try:
            self.conn = sqlite3.connect(
                db,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                uri=True)
            self.conn.text_factory = str
            self.cursor = self.conn.cursor()
            logging.info("Database connection successful.")
        except sqlite3.Error as e:
            logging.error(f"Database connection failed. Error: {e}", exc_info=True)
            raise
    
    def clear_previous_data(self):
        """
        Clear the previous data in the database.
        """

        try:
            if self.conn is not None and self.cursor is not None:
                self.cursor.execute("DELETE FROM documents")
                self.cursor.execute("DELETE FROM content")
                self.cursor.execute("DELETE FROM sqlite_sequence")
                self.conn.commit()
            else:
                logging.error("Database connection or cursor is None.")
        except sqlite3.Error as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
            if self.conn is not None:
                self.conn.rollback()
            raise
    
    def insert_data(self, organization, document_type, category, clientele, knowledge_type, language, pdf_path, content):
        """
        Allows for data insertion into the database.

        Args:
            organization: name of the organization
            document_type: type of the document
            category: category of the organization
            clientele: clientele the organization addresses
            knowledge_type: type of knowledge mobilized
            language: language of the document
            pdf_path: path to the PDF document
            content: extracted text
        """
        try:
            # Check if the required parameters are not None
            if any(param is None for param in [
                organization, document_type, category, clientele, 
                knowledge_type, language, pdf_path, content
            ]):
                raise ValueError("All parameters must be non-None")
            
            # Insert data into the documents table
            self.cursor.execute(
                "INSERT INTO documents (organization, document_type, category, clientele, knowledge_type, language, filepath) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (organization, document_type, category, clientele, knowledge_type, language, pdf_path)
            )

            # Insert data into the content table
            doc_id = self.cursor.lastrowid
            self.cursor.execute(
                "INSERT INTO content (doc_id, content) VALUES (?, ?)",
                (doc_id, content)
            )

            # Commit the changes
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)
            self.conn.rollback()
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred. Error: {e}", exc_info=True)
            raise

    def csv_to_xlsx(self, csv_path, xlsx_path):
        """
        Convert a CSV file to an XLSX file.

        Args:
            csv_path: path to the CSV file
            xlsx_path: path to the XLSX file
        """
        try:
            df = pd.read_csv(csv_path)
            logging.debug(f"CSV file {csv_path} successfully read.")
            df.to_excel(xlsx_path, index=False)
            with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            logging.info(f"XLSX file {xlsx_path} successfully created.")
        except Exception as e:
            logging.error(f"Operation failed. Error: {e}", exc_info=True)

    def fetch_all(self, lang=None):
        """
        Fetch all documents from the database.

        Args:
            lang (str): The language to filter the documents by. If None, all languages are considered.

        Returns:
            List[Tuple[int, str, str]]: Documents with their IDs, content, and language.
        """
        try:
            # Construct the SQL query
            query = """
                SELECT d.id, c.content, d.language
                FROM documents d
                JOIN content c ON d.id = c.doc_id
            """
            params = ()
            if lang:
                # Add a WHERE clause to filter by language
                query += "WHERE d.language = ?"
                params = (lang,)

            # Execute the query and return the results
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            # Log and return an empty list on error
            logging.error(f"Error fetching documents from the database: {e}", exc_info=True)
            return []

    def fetch_single(self, doc_id):
        try:
            self.cursor.execute("""
                SELECT d.idm, c.content, d.language
                FROM documents d
                JOIN content c ON d.id = c.doc_id
                WHERE d.id = ?
            """, (doc_id,))
            return self.cursor.fetchone()
        except Exception as e:
            logging.error(f"Error fetching document {doc_id} from the database: {e}", exc_info=True)
            return None
    def __del__(self):
        """
        Close the database connection.
        """
        self.conn.close()
