from pathlib import Path
from typing import Optional
import psycopg2  # type: ignore
from psycopg2.extensions import connection, cursor  # type: ignore


def setup_database(
        dbname: str = 'housing_db',
        user: str = 'housing_user',
        password: str = 'year_secure_password',
        host: str = 'localhost',
        port: str = '5432',
        schema_path: Path | str = 'sql/schema.sql') -> None:
    """
    Initialize the database by creating all necessary tables and structures.

    This function is the entry point for setting up your database from scratch.
    It connects to PostgreSQL, reads a SQL schema file containing CREATE TABLE
    statements and other database objects, and executes it to build the complete
    database structure your application needs.

    This is typically run once during initial deployment or when setting up a
    development environment. If your schema file uses "CREATE TABLE IF NOT EXISTS"
    patterns, this function is idempotent and safe to run multiple times.

    Parameters:
        dbname: The name of the PostgreSQL database to connect to
               This database must already exist (created by a DBA or deployment script)
        user: Database username with sufficient permissions to create tables
             This user needs CREATE TABLE, CREATE INDEX, and related permissions
        password: The user's password for authentication
                 In production, load this from environment variables, never hardcode
        host: The database server hostname or IP address
             Use 'localhost' for local development, or your server's address
        port: The PostgreSQL port number (standard is 5432)
             Only change this if your PostgreSQL runs on a non-standard port
        schema_path: Path to the SQL file containing your database schema
                    This file should include CREATE TABLE statements, indexes,
                    constraints, and any other structural definitions

    Returns:
        None - this function performs setup and doesn't return any value

    Usage Examples:
        # Basic usage with defaults (for local development)
        python setup_database.py

        # From another Python script with custom parameters
        from setup_database import setup_database
        setup_database(
            dbname="production_db",
            user="app_admin",
            password=os.getenv("DB_PASSWORD"),
            host="db.example.com"
        )

    Raises:
        FileNotFoundError: If the schema file doesn't exist at the specified path
        psycopg2.OperationalError: If connection to the database fails
                                  (wrong credentials, server down, etc.)
        psycopg2.ProgrammingError: If there are SQL syntax errors in the schema
        psycopg2.InsufficientPrivilege: If the user lacks permission to create tables

    Security Notes:
        The database user should have CREATE privileges but not necessarily
        SUPERUSER privileges. Follow the principle of least privilege by creating
        a separate setup user with only the permissions needed for schema creation.

        Never commit actual passwords to version control. Use environment variables:
        password=os.getenv("DB_SETUP_PASSWORD")

    Schema File Best Practices:
        Your schema.sql file should:
        - Use "CREATE TABLE IF NOT EXISTS" to make the script idempotent
        - Define all constraints (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK)
        - Create necessary indexes for query performance
        - Include comments documenting the purpose of each table
        - Be organized logically (create tables before foreign keys that reference them)
    """
    # Convert string path to Path object for robust file handling
    # This ensures cross=platform compatibility and cleaner path operations
    if isinstance(schema_path, str):
        schema_path = Path(schema_path)

    # Validate that the schema file exists before attempting connection
    # This provides a clearer error message than failing during execution
    if not schema_path.exists():
        raise FileNotFoundError(
            f'Schema file not found at: {schema_path}\n'
            f'Please ensure the SQL schema file exists at this location.'
        )

    # Establish connection to the PostgreSQL database
    # We don't use the Database class here because:
    # 1. The Database class expects tables to already exist
    # 2. We need one-time setup logic, not the operational interface
    # 3. This keeps concerns separated (setup vs. usage)
    conn: connection = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    # Create a cursor for executing SQL commands
    # The cursor is our interface for sending commands to the database
    db_cursor: cursor = conn.cursor()  # type: ignore

    try:
        # Read the entire schema file into memory as a string
        # Path.read_text() is the modern, clean way to read files
        # It handles opening, reading and closing automatically
        # The UTF-8 encoding ensures proper handling of any special characters
        schema: str = schema_path.read_text(encoding='utf-8')

        # Execute the entire schema as a single transaction
        # This provides atomicity: either all tables are created successfully,
        # or non are (if there's an error, everything rolls back)
        # THis all-or-nothing behavior prevents partial database states
        db_cursor.execute(schema)

        # Commit the transaction to make all changes permanent
        # Without this commit, all the CREATE TABLE commands would be
        # rolled back when we close the connection
        conn.commit()

        # Print success message so users know the setup completed
        # The checkmark emoji provides visual confirmation
        print('✅ Database setup complete!')
        print(f'    Tables created from {schema_path}')

    except psycopg2.Error as e:
        # If anything goes wrong during schema execution, roll back
        # This ensures we don't leave the database in a partially-created state
        conn.rollback()

        # Provide a detailed error message to help with debugging
        print(f'❌ Database setup failed!')
        print(f'    Error type: {type(e).__name__}')

        # Re-raise the exception so calling code knows setup failed
        # This is important for deployment scripts that need to detect failures
        raise

    finally:
        # Always clean up database resources, even if an error occurred
        # The finally block ensures this cleanup happens no matter what
        db_cursor.close()
        conn.close()


def verify_setup(
        dbname: str = 'housing_db',
        user: str = 'housing_user',
        password: str = 'your_secure_password',
        host: str = 'localhost',
        port: str = '5432') -> bool:
    """
        Verify that the database setup was successful by checking for key tables.

        This is a useful sanity check to run after setup_database() to confirm
        that the essential tables were created correctly. It's especially valuable
        in automated deployment pipelines where you want to verify each step.

        Parameters:
            dbname, user, password, host, port: Same as setup_database()

        Returns:
            True if all expected tables exist, False otherwise

        Example:
            setup_database()
            if verify_setup():
                print("Database is ready for use!")
            else:
                print("Setup verification failed - check for errors")
        """
    try:
        conn: connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

        db_cursor: cursor = conn.cursor()

        # Check if the key tables exist in the database
        # information_schema.tables is PostgreSQL's metadata catalog
        expected_tables: list[str] = [
            'transactions', 'categories', 'monthly_prediction']

        for table_name in expected_tables:
            db_cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
            )
                """,
                (table_name,)
            )

            exists: bool = db_cursor.fetchone()[0]
            if not exists:
                print(f'⚠️ Table "{table_name}" not found')
                db_cursor.close
                conn.close()
                return False

        db_cursor.close()
        conn.close()

        print('✅ All expected tables verified')
        return True

    except psycopg2.Error as e:
        print(f'❌ Verification failed: {e}')
        return False


if __name__ == '__main__':
    """
    Entry point when running this file directly from the command line.

    This is the standard Python pattern for making a module dual-purpose:
    - Import it to use setup_database() function in other scripts
    - Run it directly to perform initial database setup

    Usage:
        python setup_database.py

    For custom parameters, you would modify this section or create a
    separate deployment script that imports and calls setup_database()
    with your specific configuration.
    """
    # Perform the database setup with default parameters
    setup_database()

    # Optionally verify that setup succeeded
    # Comment this out if you don't want automatic verification
    verify_setup()
