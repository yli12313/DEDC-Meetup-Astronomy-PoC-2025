import csv
import psycopg2
from psycopg2 import sql
import os
from datetime import datetime

def connect_to_database():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="meetup_demo",
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def create_table(conn):
    """Create exoplanets table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS exoplanets (
        id SERIAL PRIMARY KEY,
        hostname VARCHAR(255),
        pl_name VARCHAR(255),
        pl_rade FLOAT,
        pl_masse FLOAT,
        rowupdate DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(hostname, pl_name, rowupdate)
    );
    """
    
    try:
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        print("Table 'exoplanets' created successfully")
        return True
    except psycopg2.Error as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        return False

def load_csv_to_database(conn, csv_file):
    """Load CSV data into database"""
    try:
        cur = conn.cursor()
        
        # Clear existing data (optional - remove if you want to append)
        cur.execute("TRUNCATE TABLE exoplanets RESTART IDENTITY;")
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            insert_query = """
            INSERT INTO exoplanets (hostname, pl_name, pl_rade, pl_masse, rowupdate)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (hostname, pl_name, rowupdate) DO NOTHING;
            """
            
            count = 0
            for row in csv_reader:
                # Handle empty/null values
                pl_rade = float(row['pl_rade']) if row['pl_rade'] else None
                pl_masse = float(row['pl_masse']) if row['pl_masse'] else None
                rowupdate = row['rowupdate'] if row['rowupdate'] else None
                
                cur.execute(insert_query, (
                    row['hostname'],
                    row['pl_name'],
                    pl_rade,
                    pl_masse,
                    rowupdate
                ))
                count += 1
            
            conn.commit()
            cur.close()
            print(f"Successfully loaded {count} records into database")
            return count
            
    except Exception as e:
        print(f"Error loading data: {e}")
        conn.rollback()
        return 0

def get_latest_csv_file():
    """Find the most recent TESS exoplanets CSV file"""
    csv_files = [f for f in os.listdir('.') if f.startswith('tess_exoplanets_') and f.endswith('.csv')]
    if not csv_files:
        return None
    
    # Sort by filename (which includes timestamp)
    csv_files.sort(reverse=True)
    return csv_files[0]

def main():
    """Main function to load CSV data into database"""
    print("Starting database load process...")
    
    # Find latest CSV file
    csv_file = get_latest_csv_file()
    if not csv_file:
        print("No TESS exoplanets CSV file found")
        return
    
    print(f"Using CSV file: {csv_file}")
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        # Create table
        if not create_table(conn):
            return
        
        # Load data
        records_loaded = load_csv_to_database(conn, csv_file)
        
        if records_loaded > 0:
            print(f"Database load completed successfully!")
            print(f"Total records: {records_loaded}")
        else:
            print("No records were loaded")
            
    finally:
        conn.close()

if __name__ == "__main__":
    main()