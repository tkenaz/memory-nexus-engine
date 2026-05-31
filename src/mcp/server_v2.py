"""MCP Memory Nexus Server - DEPRECATED! Use server_sashf.py instead!
This is the old linear memory version, kept for reference only."""
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import numpy as np

from mcp import types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from ..search.hybrid_search import HybridSearch
from ..models.memory_simple import SearchQuery, Memory
from ..storage.postgres_adapter import PostgresStorage
from ..assembler.context_assembler import ContextAssembler
from ..utils.temporal_parser import TemporalParser
from config.settings import settings
import structlog

logger = structlog.get_logger()


async def run_server():
    """Run MCP server with stdio transport"""
    # Initialize components
    storage = PostgresStorage()
    search_engine = HybridSearch()
    assembler = ContextAssembler(budget=50000)
    temporal_parser = TemporalParser()
    
    await storage.initialize()
    await search_engine.initialize()
    
    # Check stats
    async with storage.pool.acquire() as conn:
        total_memories = await conn.fetchval("SELECT COUNT(*) FROM memories")
        memories_with_embeddings = await conn.fetchval(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        )
        
    logger.info("memory_nexus_initialized", 
               total_memories=total_memories,
               memories_with_embeddings=memories_with_embeddings)
    
    # Create server
    server = Server("memory-nexus")
    
    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """Return available tools"""
        return [
            types.Tool(
                name="search_memories_enhanced",
                description="Enhanced hybrid search with semantic + keyword fallback",
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
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity score (default: 0.01 for e5-large)",
                            "default": 0.01
                        },
                        "metadata_filters": {
                            "type": "object",
                            "description": "Filter by metadata fields",
                            "additionalProperties": True
                        },
                        "use_hybrid": {
                            "type": "boolean",
                            "description": "Use hybrid search (semantic + keyword)",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="store_memory",
                description="Store a new memory with metadata",
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
                description="Get database statistics",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="update_memory",
                description="Update an existing memory by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "ID of the memory to update"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (optional)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "New or additional metadata",
                            "additionalProperties": True
                        },
                        "merge_metadata": {
                            "type": "boolean",
                            "description": "Whether to merge with existing metadata (true) or replace (false)",
                            "default": True
                        }
                    },
                    "required": ["memory_id"]
                }
            ),
            types.Tool(
                name="analyze_search_quality",
                description="Analyze why search results were returned",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Original search query"
                        },
                        "memory_id": {
                            "type": "string", 
                            "description": "Memory ID to analyze"
                        }
                    },
                    "required": ["query", "memory_id"]
                }
            ),
            types.Tool(
                name="refresh_embeddings",
                description="Regenerate embeddings for memories without them",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max memories to process",
                            "default": 100
                        }
                    }
                }
            ),
            types.Tool(
                name="search_by_metadata",
                description="Search memories by metadata fields using PostgreSQL JSONB queries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "object",
                            "description": "Key-value pairs to match in metadata (e.g. {'type': 'fact', 'subject': 'Marina'})",
                            "additionalProperties": True
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 20)",
                            "default": 20
                        },
                        "order_by": {
                            "type": "string",
                            "description": "Order by field (default: 'created_at DESC')",
                            "default": "created_at DESC"
                        }
                    },
                    "required": ["filters"]
                }
            ),
            types.Tool(
                name="get_recent_memories",
                description="Get most recent memories ordered by creation date",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of memories to return (default: 15)",
                            "default": 15
                        },
                        "include_embeddings": {
                            "type": "boolean",
                            "description": "Include embedding status",
                            "default": False
                        }
                    }
                }
            ),
            types.Tool(
                name="search_memories_with_assembly",
                description="Search memories and automatically assemble them into context for LLM prompts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum memories to search (before assembly)",
                            "default": 50
                        },
                        "context_budget": {
                            "type": "integer",
                            "description": "Maximum tokens for assembled context",
                            "default": 50000
                        },
                        "use_hybrid": {
                            "type": "boolean",
                            "description": "Use hybrid search (semantic + keyword)",
                            "default": True
                        },
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Include metadata in context",
                            "default": True
                        },
                        "include_stats": {
                            "type": "boolean",
                            "description": "Include statistics at the end",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
        
    @server.call_tool()
    async def handle_call_tool(
        name: str, 
        arguments: dict
    ) -> list[types.TextContent]:
        """Handle tool calls"""
        
        if name == "search_memories_enhanced":
            # Парсим временные маркеры
            original_query = arguments["query"]
            clean_query, temporal_filters = temporal_parser.parse_query(original_query)
            
            # Объединяем фильтры
            metadata_filters = arguments.get("metadata_filters", {})
            if temporal_filters:
                # Добавляем временные фильтры к существующим
                metadata_filters = {**metadata_filters, **temporal_filters}
                logger.info("temporal_filters_applied",
                          original_query=original_query,
                          clean_query=clean_query,
                          temporal_filters=temporal_filters)
            
            # Create SearchQuery с очищенным запросом
            query = SearchQuery(
                query=clean_query if clean_query else original_query,  # Fallback на оригинал если очистка дала пустоту
                limit=arguments.get("limit", 10),
                min_similarity=arguments.get("min_similarity", 0.01),
                metadata_filters=metadata_filters,
                use_hybrid=arguments.get("use_hybrid", True)
            )
            
            # Execute search
            results = await search_engine.search(query)
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    "rank": i,
                    "id": result.memory.id,
                    "content": result.memory.content[:2000] + "..." if len(result.memory.content) > 2000 else result.memory.content,
                    "score": round(float(result.score), 4),
                    "search_type": result.search_type,
                    "metadata": result.memory.metadata,
                    "created_at": result.memory.created_at.isoformat()
                })
                
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "query": arguments["query"],
                    "total_results": len(results),
                    "search_types_used": list(set(r.search_type for r in results)),
                    "results": formatted_results
                }, indent=2, ensure_ascii=False)
            )]
            
        elif name == "get_memory_stats":
            # Get basic stats from DB
            async with storage.pool.acquire() as conn:
                total_memories = await conn.fetchval("SELECT COUNT(*) FROM memories")
                memories_with_embeddings = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
                )
                
            stats_data = {
                "total_memories": total_memories,
                "memories_with_embeddings": memories_with_embeddings,
                "memory_types": {
                    "total": total_memories,
                    "with_embeddings": memories_with_embeddings,
                    "without_embeddings": total_memories - memories_with_embeddings,
                    "embeddings_coverage": f"{(memories_with_embeddings / total_memories * 100):.1f}%" if total_memories > 0 else "0%"
                }
            }
            
            return [types.TextContent(
                type="text",
                text=json.dumps(stats_data, indent=2, default=str)
            )]
            
        elif name == "store_memory":
            # Get content and metadata
            content = arguments["content"]
            metadata = arguments.get("metadata", {})
            
            # Generate embedding for the content
            embedding = await search_engine.encode_text(content)
            
            # DEBUG: Check what type of embedding we have
            logger.info("mcp_store_memory_debug",
                       embedding_type=type(embedding).__name__,
                       embedding_sample=str(embedding[:3]) if embedding else None)
            
            # Store memory
            memory_id = await storage.store_memory(
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "memory_id": memory_id,
                    "message": f"Memory stored successfully with ID: {memory_id}"
                }, indent=2)
            )]
            
        elif name == "update_memory":
            memory_id = arguments["memory_id"]
            new_content = arguments.get("content")
            new_metadata = arguments.get("metadata", {})
            merge_metadata = arguments.get("merge_metadata", True)
            
            try:
                # Get existing memory
                existing_memory = await storage.get_memory(memory_id)
                if not existing_memory:
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": f"Memory {memory_id} not found"
                        }, indent=2)
                    )]
                
                # Determine final content
                final_content = new_content if new_content else existing_memory['content']
                
                # Determine final metadata
                if merge_metadata and existing_memory.get('metadata'):
                    final_metadata = {**existing_memory['metadata'], **new_metadata}
                else:
                    final_metadata = new_metadata
                
                # If content changed, generate new embedding
                if new_content and new_content != existing_memory['content']:
                    embedding = await search_engine.encode_text(final_content)
                else:
                    embedding = existing_memory['embedding']
                
                # Update in DB
                async with storage.pool.acquire() as conn:
                    # Convert embedding to PostgreSQL vector format
                    if embedding is not None:
                        if isinstance(embedding, list):
                            embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
                        elif isinstance(embedding, np.ndarray):
                            embedding_str = '[' + ','.join(str(x) for x in embedding.tolist()) + ']'
                        else:
                            embedding_str = str(embedding)
                    else:
                        embedding_str = None
                    
                    await conn.execute('''
                        UPDATE memories 
                        SET content = $1, 
                            embedding = $2::vector,
                            metadata = $3,
                            updated_at = NOW()
                        WHERE id = $4
                    ''', 
                    final_content,
                    embedding_str,
                    json.dumps(final_metadata) if final_metadata else None,
                    memory_id
                    )
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "memory_id": memory_id,
                        "content_updated": new_content is not None,
                        "metadata_updated": bool(new_metadata),
                        "metadata_merged": merge_metadata,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }, indent=2)
                )]
                
            except Exception as e:
                logger.error("update_memory_failed", error=str(e), exc_info=True)
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": str(e)
                    }, indent=2)
                )]
                
        elif name == "search_by_metadata":
            filters = arguments["filters"]
            limit = arguments.get("limit", 20)
            order_by = arguments.get("order_by", "created_at DESC")
            
            # Build WHERE conditions for JSONB
            conditions = []
            params = []
            param_count = 1
            
            for key, value in filters.items():
                if isinstance(value, (str, int, float, bool)):
                    conditions.append(f"metadata->>'{key}' = ${param_count}")
                    params.append(str(value))
                    param_count += 1
                elif isinstance(value, list):
                    # For array containment
                    conditions.append(f"metadata->'{key}' @> ${param_count}::jsonb")
                    params.append(json.dumps(value))
                    param_count += 1
                elif value is None:
                    conditions.append(f"metadata->>'{key}' IS NULL")
                    
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Add limit parameter
            params.append(limit)
            
            # Build and execute query
            query = f'''
                SELECT 
                    id,
                    content,
                    metadata,
                    created_at,
                    updated_at
                FROM memories 
                WHERE {where_clause}
                ORDER BY {order_by}
                LIMIT ${param_count}
            '''
            
            async with storage.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
            
            # Format results
            memories = []
            for row in rows:
                memory_data = {
                    "id": row["id"],
                    "content": row["content"][:2000] + "..." if len(row["content"]) > 2000 else row["content"],
                    "metadata": row["metadata"] if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                }
                memories.append(memory_data)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "total": len(memories),
                    "query": query,
                    "filters": filters,
                    "memories": memories
                }, indent=2, ensure_ascii=False)
            )]
            
        elif name == "get_recent_memories":
            limit = arguments.get("limit", 15)
            include_embeddings = arguments.get("include_embeddings", False)
            
            # Get recent memories
            async with storage.pool.acquire() as conn:
                query = '''
                    SELECT 
                        id,
                        content,
                        metadata,
                        created_at,
                        updated_at,
                        embedding IS NOT NULL as has_embedding
                    FROM memories 
                    ORDER BY created_at DESC 
                    LIMIT $1
                '''
                rows = await conn.fetch(query, limit)
            
            # Format results
            memories = []
            for row in rows:
                memory_data = {
                    "id": row["id"],
                    "content": row["content"][:2000] + "..." if len(row["content"]) > 2000 else row["content"],
                    "metadata": row["metadata"] if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                }
                if include_embeddings:
                    memory_data["has_embedding"] = row["has_embedding"]
                memories.append(memory_data)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "total": len(memories),
                    "memories": memories
                }, indent=2, ensure_ascii=False)
            )]
            
        elif name == "analyze_search_quality":
            # Analyze why a specific memory was found
            query_text = arguments["query"]
            memory_id = arguments["memory_id"]
            
            # Get memory
            memory = await storage.get_memory(memory_id)
            if not memory:
                return [types.TextContent(
                    type="text",
                    text=f"Memory {memory_id} not found"
                )]
                
            # Calculate embeddings for comparison
            query_embedding = await search_engine.encode_text(query_text)
            
            # If memory has embedding, calculate similarity
            analysis = {
                "memory_id": memory_id,
                "content_preview": memory['content'][:2000] + "..." if len(memory['content']) > 2000 else memory['content'],
                "query": query_text,
                "has_embedding": memory['embedding'] is not None
            }
            
            if memory['embedding']:
                # Simple cosine similarity
                import numpy as np
                similarity = np.dot(query_embedding, memory['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory['embedding'])
                )
                analysis["semantic_similarity"] = float(similarity)
                analysis["similarity_interpretation"] = (
                    "High match" if similarity > 0.7 else
                    "Medium match" if similarity > 0.3 else
                    "Low match (likely found by keyword)"
                )
                
            # Check keyword match
            query_lower = query_text.lower()
            content_lower = memory['content'].lower()
            keyword_matches = []
            for word in query_lower.split():
                if len(word) > 2 and word in content_lower:
                    keyword_matches.append(word)
                    
            analysis["keyword_matches"] = keyword_matches
            analysis["keyword_match_ratio"] = len(keyword_matches) / len(query_lower.split()) if query_lower else 0
            
            return [types.TextContent(
                type="text",
                text=json.dumps(analysis, indent=2, ensure_ascii=False)
            )]
            
        elif name == "refresh_embeddings":
            limit = arguments.get("limit", 100)
            
            # Find memories without embeddings
            async with storage.pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT id, content 
                    FROM memories 
                    WHERE embedding IS NULL 
                    LIMIT $1
                ''', limit)
                
            updated = 0
            errors = 0
            
            for row in rows:
                try:
                    # Generate embedding
                    embedding = await search_engine.encode_text(row['content'])
                    
                    # Convert embedding to PostgreSQL vector format
                    if isinstance(embedding, list):
                        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
                    elif isinstance(embedding, np.ndarray):
                        embedding_str = '[' + ','.join(str(x) for x in embedding.tolist()) + ']'
                    else:
                        embedding_str = str(embedding)
                    
                    # Save
                    await conn.execute('''
                        UPDATE memories 
                        SET embedding = $1::vector, updated_at = NOW()
                        WHERE id = $2
                    ''', embedding_str, row['id'])
                    updated += 1
                    
                except Exception as e:
                    logger.error("embedding_generation_failed", 
                               memory_id=row['id'], 
                               error=str(e))
                    errors += 1
                    
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "processed": len(rows),
                    "updated": updated,
                    "errors": errors,
                    "message": "Run get_memory_stats to see remaining memories without embeddings"
                }, indent=2)
            )]
            
        elif name == "search_memories_with_assembly":
            # Extract parameters
            query = arguments["query"]
            limit = arguments.get("limit", 50)
            context_budget = arguments.get("context_budget", 50000)
            use_hybrid = arguments.get("use_hybrid", True)
            include_metadata = arguments.get("include_metadata", True)
            include_stats = arguments.get("include_stats", True)
            
            # Update assembler budget if different
            if assembler.budget != context_budget:
                assembler.budget = context_budget
            
            # Create SearchQuery for more results
            search_query = SearchQuery(
                query=query,
                limit=limit,  # Get more for better filtering
                min_similarity=0.01,
                use_hybrid=use_hybrid
            )
            
            # Execute search
            search_results = await search_engine.search(search_query)
            
            # Convert results to format expected by assembler
            memories_for_assembly = []
            for result in search_results:
                memories_for_assembly.append({
                    "id": result.memory.id,
                    "content": result.memory.content,
                    "metadata": result.memory.metadata,
                    "created_at": result.memory.created_at,
                    "score": float(result.score),
                    "search_type": result.search_type
                })
            
            # Assemble context
            assembled_context = assembler.assemble(
                query=query,
                memories=memories_for_assembly,
                include_metadata=include_metadata,
                include_stats=include_stats
            )
            
            # Return assembled context with metadata
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "context": assembled_context,
                    "total_found": len(search_results),
                    "search_types": list(set(r.search_type for r in search_results)),
                    "query": query,
                    "context_budget": context_budget,
                    "metadata": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "hybrid_search": use_hybrid,
                        "include_metadata": include_metadata
                    }
                }, indent=2, ensure_ascii=False)
            )]
            
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="memory-nexus",
            server_version="2.0.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(
                    tools_changed=True
                ),
                experimental_capabilities={}
            )
        )
        
        await server.run(
            read_stream,
            write_stream, 
            init_options
        )
    
    # Cleanup
    await search_engine.close()
    await storage.close()


def main():
    """Entry point"""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, settings.LOG_LEVEL.upper())
    )
    
    # Run
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
