"""MCP Memory Nexus Server with SASH-F capabilities"""
import asyncio
import json
import sys
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from mcp import types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from ..storage.postgres_enhanced import EnhancedPostgresStorage
from ..models.memory_simple import SearchQuery, Memory
from ..assembler.context_assembler import ContextAssembler
from ..utils.temporal_parser import TemporalParser
from config.settings import settings

# Disable structlog stdout output for MCP
logging.basicConfig(level=logging.ERROR, handlers=[])


async def run_server():
    """Run MCP server with SASH-F enhanced storage"""
    # Initialize enhanced storage with SASH-F
    storage = EnhancedPostgresStorage()
    assembler = ContextAssembler(budget=50000)
    temporal_parser = TemporalParser()
    
    await storage.initialize()
    
    # Get initial stats
    stats = await storage.get_graph_stats()
    
    # Create server
    server = Server("memory-nexus-sashf")
    
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="search_memories_enhanced",
                description="Enhanced hybrid search with semantic + keyword fallback + SASH-F structure",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default: 10)",
                            "default": 10
                        },
                        "use_graph": {
                            "type": "boolean",
                            "description": "Use graph expansion (default: true)",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="store_memory",
                description="Store a new memory with SASH-F structure extraction",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content to store"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata (type, tags, etc)",
                            "additionalProperties": True
                        }
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="get_memory_stats",
                description="Get database statistics including graph and patterns",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="get_patterns",
                description="Get detected patterns in memories",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_occurrences": {
                            "type": "integer",
                            "description": "Minimum pattern occurrences",
                            "default": 3
                        }
                    }
                }
            ),
            types.Tool(
                name="get_recent_memories",
                description="Get most recent memories by date (no semantic search, just chronological order)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of memories to return (default: 10)",
                            "default": 10
                        }
                    }
                }
            )
        ]
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle tool calls"""
        try:
            if name == "search_memories_enhanced":
                # search_with_graph already handles temporal parsing internally!
                results = await storage.search_with_graph(
                    query=arguments["query"],  # Pass original query with temporal markers
                    limit=arguments.get("limit", 10),
                    use_graph=arguments.get("use_graph", True),
                    graph_depth=2
                )
                

                # Format results
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "id": r["id"],
                        "content": r["content"],
                        "similarity": float(r.get("similarity", 0)),
                        "sashf": r.get("sashf", {}),
                        "metadata": r.get("metadata", {})
                    })
                
                # Add debug info if no results
                response = formatted_results
                if not formatted_results:
                    response = {
                        "results": [],
                        "debug": {
                            "query": arguments["query"],
                            "raw_results_count": len(results),
                            "message": "No results found - check if embeddings exist and model loaded"
                        }
                    }
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(response, indent=2, ensure_ascii=False)
                )]
            
            elif name == "store_memory":
                content = arguments["content"]
                from ..core import chunker

                # For short content: generate embedding here
                # For long content: store_memory_enhanced will generate per-chunk embeddings
                embedding = None
                if not chunker.needs_chunking(content):
                    embedding_list = await storage.encode_text(content)
                    embedding = embedding_list

                memory_id = await storage.store_memory_enhanced(
                    content=content,
                    embedding=embedding,
                    metadata=arguments.get("metadata", {})
                )
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "memory_id": memory_id,
                        "message": f"Memory stored with SASH-F structure: {memory_id}"
                    }, ensure_ascii=False)
                )]
            
            elif name == "get_memory_stats":
                stats = await storage.get_graph_stats()
                
                # Get pattern stats too
                patterns = await storage.get_patterns(min_occurrences=1)
                stats["total_patterns"] = len(patterns)
                stats["top_patterns"] = [
                    {
                        "key": p["pattern_key"],
                        "occurrences": p["occurrences"],
                        "confidence": p["confidence"]
                    }
                    for p in patterns[:5]
                ]
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2, ensure_ascii=False)
                )]
            
            elif name == "get_patterns":
                patterns = await storage.get_patterns(
                    min_occurrences=arguments.get("min_occurrences", 3)
                )
                
                formatted_patterns = []
                for p in patterns:
                    formatted_patterns.append({
                        "pattern_key": p["pattern_key"],
                        "occurrences": p["occurrences"],
                        "confidence": p["confidence"],
                        "actors": p.get("actors", []),
                        "act": p.get("act"),
                        "manner": p.get("manner")
                    })
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(formatted_patterns, indent=2, ensure_ascii=False)
                )]
            
            elif name == "get_recent_memories":
                limit = arguments.get("limit", 10)

                async with storage.pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id, content, metadata, sashf, created_at
                        FROM memories
                        WHERE parent_id IS NULL
                        ORDER BY created_at DESC
                        LIMIT $1
                    """, limit)

                memories = []
                for row in rows:
                    memories.append({
                        "id": row["id"],
                        "content": row["content"][:2000] + "..." if len(row["content"]) > 2000 else row["content"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "sashf": json.loads(row["sashf"]) if row["sashf"] else {},
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None
                    })

                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "total": len(memories),
                        "memories": memories
                    }, indent=2, ensure_ascii=False)
                )]

            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Unknown tool: {name}"
                    })
                )]
                
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": str(e),
                    "tool": name
                }, ensure_ascii=False)
            )]
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="memory-nexus-sashf",
                server_version="3.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


def main():
    """Main entry point"""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
