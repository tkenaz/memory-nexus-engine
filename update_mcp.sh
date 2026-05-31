#!/bin/bash
# Update MCP to latest version in memory-nexus-v2

cd /Users/mvyshhnyvetska/Desktop/Claude/memory-nexus-v2

echo "=== Updating MCP packages ==="

source .venv/bin/activate

# Показываем текущие версии
echo "Current versions:"
pip list | grep -E "(mcp|pydantic|anyio)"

# Обновляем MCP и связанные пакеты
echo -e "\nUpdating MCP..."
pip install --upgrade mcp

# Обновляем pydantic если нужно
echo -e "\nUpdating pydantic..."
pip install --upgrade pydantic pydantic-core

# Обновляем anyio
echo -e "\nUpdating anyio..."
pip install --upgrade anyio

echo -e "\nNew versions:"
pip list | grep -E "(mcp|pydantic|anyio)"

# Тестируем импорты
echo -e "\n=== Testing imports ==="
python -c "
from mcp.server import Server
from mcp import types
from pydantic import BaseModel
print('✓ MCP imports OK')
print('✓ Pydantic imports OK')
"

echo -e "\nDone! Try restarting Claude now."
