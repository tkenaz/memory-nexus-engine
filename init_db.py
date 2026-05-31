#!/usr/bin/env python3
"""
Initialize Memory Nexus database with all migrations.

Usage:
    python init_db.py

Environment:
    Requires DATABASE_URL in .env file or environment variable
"""
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import asyncpg

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def get_db_connection():
    """Create database connection from DATABASE_URL"""
    if not DATABASE_URL:
        print("❌ ERROR: DATABASE_URL not found in environment")
        print("   Please create .env file with DATABASE_URL")
        print("   Example: DATABASE_URL=postgresql://postgres:password@localhost:5432/memory_nexus")
        sys.exit(1)

    # Handle both postgresql:// and postgres:// schemes
    dsn = DATABASE_URL.replace("postgresql://", "").replace("postgres://", "")

    # Parse DSN
    try:
        # Format: username:password@host:port/database
        if "@" not in dsn:
            print(f"❌ Invalid DATABASE_URL format: {DATABASE_URL}")
            print("   Expected: postgresql://username:password@host:port/database")
            sys.exit(1)

        auth, location = dsn.split("@", 1)
        username, password = auth.split(":", 1) if ":" in auth else (auth, "")

        if "/" in location:
            hostport, database = location.split("/", 1)
        else:
            hostport = location
            database = "postgres"

        if ":" in hostport:
            host, port = hostport.split(":", 1)
            port = int(port)
        else:
            host = hostport
            port = 5432

        print(f"🔌 Connecting to PostgreSQL at {host}:{port}/{database}")

        conn = await asyncpg.connect(
            user=username,
            password=password,
            host=host,
            port=port,
            database=database
        )

        print("✅ Connected to database")
        return conn

    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print(f"   DATABASE_URL: {DATABASE_URL}")
        sys.exit(1)


async def run_migration(conn, migration_file: Path):
    """Execute a single migration file"""
    print(f"📄 Running migration: {migration_file.name}")

    try:
        sql = migration_file.read_text()
        await conn.execute(sql)
        print(f"   ✅ {migration_file.name} completed")
        return True
    except Exception as e:
        print(f"   ❌ {migration_file.name} failed: {e}")
        return False


async def main():
    """Run all migrations in order"""
    print("=" * 70)
    print("Memory Nexus - Database Initialization")
    print("=" * 70)
    print()

    # Check migrations directory
    if not MIGRATIONS_DIR.exists():
        print(f"❌ Migrations directory not found: {MIGRATIONS_DIR}")
        sys.exit(1)

    # Get all SQL migration files in order
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not migration_files:
        print("❌ No migration files found in migrations/")
        sys.exit(1)

    print(f"📦 Found {len(migration_files)} migration(s):")
    for mf in migration_files:
        print(f"   - {mf.name}")
    print()

    # Connect to database
    conn = await get_db_connection()

    try:
        # Run migrations
        print("🚀 Starting migrations...")
        print()

        success_count = 0
        for migration_file in migration_files:
            if await run_migration(conn, migration_file):
                success_count += 1
            else:
                print()
                print("❌ Migration failed, stopping")
                sys.exit(1)

        print()
        print("=" * 70)
        print(f"✅ All migrations completed successfully ({success_count}/{len(migration_files)})")
        print("=" * 70)
        print()

        # Show stats
        stats = await conn.fetchrow("SELECT * FROM memory_stats_basic")
        if stats:
            print("📊 Database stats:")
            print(f"   Total memories: {stats['total_memories']}")
            print(f"   With embeddings: {stats['memories_with_embeddings']}")
            print(f"   Table size: {stats['table_size']}")

        print()
        print("🎉 Memory Nexus database is ready!")
        print()
        print("Next steps:")
        print("  1. Test connection: python test_mcp_final.py")
        print("  2. Add to Claude Desktop config (see INSTALL.md)")
        print("  3. Restart Claude Desktop")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
