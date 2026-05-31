#!/bin/bash

# Run SASH-F migration for Memory Nexus
# Date: 2025-10-03

echo "🚀 Starting SASH-F & Graph migration for Memory Nexus..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Load environment
source .env

# Build DSN
PG_DSN="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${PG_DB}"

# Check PostgreSQL connection
echo "📊 Checking database connection..."
echo "   Using database: ${PG_DB}"

psql "$PG_DSN" -c "SELECT version();" > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Error: Cannot connect to PostgreSQL!"
    echo "   Connection string: $PG_DSN"
    echo "   Please check your credentials"
    exit 1
fi

echo "✅ Database connection OK"

# Run migration
echo "🔧 Running migration..."
psql "$PG_DSN" -f migrations/002_add_sashf_and_graph.sql

if [ $? -eq 0 ]; then
    echo "✅ Migration completed successfully!"
    
    # Show stats
    echo ""
    echo "📈 Current database stats:"
    psql "$PG_DSN" -c "SELECT * FROM memory_stats;" 2>/dev/null || echo "Stats view will be available after first memory is stored"
    
    echo ""
    echo "🎉 SASH-F structure and graph capabilities are now ready!"
    echo ""
    echo "New features available:"
    echo "  • Structured memory with actors, actions, manner, affect"
    echo "  • Automatic temporal and semantic linking"
    echo "  • Graph traversal up to N hops"
    echo "  • Pattern detection and promotion"
    echo "  • Enhanced search with structure matching"
else
    echo "❌ Migration failed! Check the error messages above."
    exit 1
fi
