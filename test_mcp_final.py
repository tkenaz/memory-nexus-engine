#!/usr/bin/env python3
"""
Final test for MCP Memory Nexus
Тест для проверки что все работает идеально
"""
import asyncio
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.postgres_enhanced import EnhancedPostgresStorage


async def test_all_features():
    """Полный тест всех функций"""
    storage = EnhancedPostgresStorage()
    await storage.initialize()
    
    print("=" * 60)
    print("🧪 MEMORY NEXUS FINAL TEST")
    print("=" * 60)
    
    # 1. Test stats
    print("\n📊 DATABASE STATS:")
    stats = await storage.get_graph_stats()
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  With embeddings: {stats.get('memories_with_embeddings', 0)}")
    print(f"  Total links: {stats.get('total_links', 0)}")
    print(f"  Patterns: {stats.get('patterns_detected', 0)}")
    
    # 2. Test semantic search
    print("\n🔍 SEMANTIC SEARCH TEST:")
    test_queries = [
        "projects",
        "someone tired",
        "a quiet evening",
        "robot project",
        "Sonnet Cursor"
    ]
    
    for query in test_queries:
        results = await storage.search_with_graph(
            query=query,
            limit=3,
            use_graph=False  # Без графа для чистоты теста
        )
        print(f"\n  Query: '{query}'")
        print(f"  Found: {len(results)} results")
        if results:
            top_score = results[0].get('similarity', 0)
            print(f"  Top score: {top_score:.4f}")
            print(f"  Best match: {results[0]['content'][:60]}...")
    
    # 3. Test graph expansion
    print("\n\n🕸️ GRAPH EXPANSION TEST:")
    results = await storage.search_with_graph(
        query="user assistant work",
        limit=5,
        use_graph=True,
        graph_depth=2
    )
    
    print(f"  With graph: {len(results)} results")
    
    # Check for different link types in results
    link_types = set()
    async with storage.pool.acquire() as conn:
        for r in results[:3]:
            links = await conn.fetch("""
                SELECT DISTINCT link_type 
                FROM memory_links 
                WHERE from_id = $1 OR to_id = $1
            """, r['id'])
            for l in links:
                link_types.add(l['link_type'])
    
    print(f"  Link types used: {', '.join(link_types)}")
    
    # 4. Test patterns
    print("\n\n🔄 PATTERN TEST:")
    patterns = await storage.get_patterns(min_occurrences=3)
    print(f"  Found {len(patterns)} patterns (min 3 occurrences)")
    if patterns:
        for p in patterns[:3]:
            print(f"    • {p['pattern_key']} ({p['occurrences']} times)")
    
    # 5. Store new memory test
    print("\n\n💾 STORE NEW MEMORY TEST:")
    test_content = f"Test memory from MCP final test at {datetime.now()}"
    
    # Generate embedding
    test_embedding = await storage.encode_text(test_content)
    
    # Store
    memory_id = await storage.store_memory_enhanced(
        content=test_content,
        embedding=test_embedding,
        metadata={"source": "test", "test": True}
    )
    
    print(f"  Stored: {memory_id}")
    
    # Verify it's searchable
    await asyncio.sleep(1)  # Wait for async linking
    
    verify_results = await storage.search_with_graph(
        query="MCP final test",
        limit=1,
        use_graph=False
    )
    
    if verify_results and verify_results[0]['id'] == memory_id:
        print(f"  ✅ Verified: Found in search!")
        print(f"     Score: {verify_results[0].get('similarity', 0):.4f}")
    else:
        print(f"  ⚠️ Not found in search yet")
    
    # Clean up test memory
    async with storage.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)
        print(f"  🧹 Cleaned up test memory")
    
    await storage.close()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    print("Starting Memory Nexus final test...")
    asyncio.run(test_all_features())
