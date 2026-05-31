#!/usr/bin/env python3
"""Run Memory Nexus SASH-F MCP Server"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mcp.server_sashf import main

if __name__ == "__main__":
    main()
