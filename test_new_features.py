#!/usr/bin/env python3
"""Test new features: recency boost and query intent detection"""

import asyncio
import sys
from datetime import datetime, timedelta
from src.storage.postgres_enhanced import EnhancedPostgresStorage
from src.core.query_intent import QueryIntentDetector

async def test_query_intent():
    """Test query intent detection"""
    print("\n🎯 TESTING QUERY INTENT DETECTION")
    print("=" * 50)

    detector = QueryIntentDetector()

    test_queries = [
        "последние разговоры с Алисой",    # TEMPORAL (+ PERSON if configured)
        "memory-nexus проект",             # PROJECT
        "Алиса устала вчера",              # EMOTIONAL + TEMPORAL
        "баг в memory-nexus",              # TECHNICAL + PROJECT
        "$4000 на оборудование",           # FACTUAL
        "проект последние новости"         # TEMPORAL
    ]

    for query in test_queries:
        intent, context = detector.detect_intent(query)
        enhanced = detector.disambiguate_query(query, intent, context)

        print(f"\n📝 Query: '{query}'")
        print(f"   Intent: {intent.value}")
        print(f"   Entities: {[k for k, v in context['entities'].items() if v]}")
        if enhanced != query:
            print(f"   Enhanced: '{enhanced}'")
        if context['boost_factors']:
            print(f"   Boosts: {context['boost_factors']}")


async def test_search_with_features():
    """Test search with recency boost and intent detection"""
    print("\n🔍 TESTING ENHANCED SEARCH")
    print("=" * 50)

    storage = EnhancedPostgresStorage()
    await storage.initialize()  # Initialize pool

    test_queries = [
        ("последние", "Should prioritize recent memories"),
        ("memory-nexus", "Should detect the project entity"),
        ("вчера", "Should find yesterday's memories"),
    ]

    for query, description in test_queries:
        print(f"\n🔎 Query: '{query}' - {description}")

        results = await storage.search_with_graph(
            query=query,
            limit=3,
            use_graph=False  # Disable graph for cleaner test
        )

        for i, r in enumerate(results, 1):
            print(f"\n  {i}. [{r['similarity']:.3f}] {r['content'][:60]}...")
            if 'recency_boost' in r:
                print(f"     Recency: {r['recency_boost']:.3f}, Days old: {r.get('days_old', 'N/A')}")
            if 'intent' in r:
                print(f"     Intent: {r['intent']}")

    # Storage will auto-close pool


async def test_recency_comparison():
    """Compare search results with and without recency boost"""
    print("\n⚖️  TESTING RECENCY BOOST EFFECT")
    print("=" * 50)

    storage = EnhancedPostgresStorage()
    await storage.initialize()  # Initialize pool

    # Add test memories with different dates
    test_memories = [
        ("Пользователь работает над новой фичей recency boost", {"type": "test", "recent": True}),
        ("Ассистент помогает с query intent detection", {"type": "test", "recent": True})
    ]

    print("\n💾 Adding test memories...")
    for content, metadata in test_memories:
        memory_id = await storage.store_memory_enhanced(
            content=content,
            embedding=await storage.encode_text(content),
            metadata=metadata
        )
        print(f"   Stored: {memory_id[:20]}... - {content[:40]}...")

    # Wait a bit for async operations
    await asyncio.sleep(1)

    # Search for recent test memories
    print("\n🔍 Searching for 'test recency boost'...")
    results = await storage.search_with_graph(
        query="test recency boost",
        limit=5
    )

    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['similarity']:.3f}] {r['content'][:60]}...")
        if r.get('days_old') is not None:
            print(f"     Days old: {r['days_old']}, Recency boost: {r.get('recency_boost', 0):.3f}")

    # Storage will auto-close pool


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 TESTING NEW MEMORY NEXUS FEATURES")
    print("=" * 60)

    try:
        await test_query_intent()
        await test_search_with_features()
        await test_recency_comparison()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())