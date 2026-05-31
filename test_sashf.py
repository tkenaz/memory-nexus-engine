#!/usr/bin/env python3
"""Test SASH-F enhanced memory system"""

import asyncio
import json
from datetime import datetime
from pprint import pprint

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.postgres_enhanced import EnhancedPostgresStorage
from src.core.sashf_parser import SASHFParser
from config.settings import settings


async def test_sashf_parser():
    """Test SASH-F parser"""
    print("\n🔬 Testing SASH-F Parser...")
    
    parser = SASHFParser()
    
    test_cases = [
        "Алиса устало вздохнула после долгого рабочего дня",
        "Ассистент внимательно анализирует новую архитектуру памяти",
        "Кот громко мяукает, требуя внимания",
        "Боб спокойно работает в соседней комнате",
        "Вчера мы обсуждали план нового проекта",
    ]
    
    for text in test_cases:
        print(f"\n📝 Text: {text}")
        result = parser.parse(text)
        print(f"  Actors: {result['actors']}")
        print(f"  Act: {result['act']}")
        print(f"  Manner: {result['manner']}")
        if result['affect']:
            print(f"  Affect: V={result['affect'].valence:.2f}, "
                  f"A={result['affect'].arousal:.2f}, "
                  f"D={result['affect'].dominance:.2f}")
        print(f"  Qualia: {result['qualia']}")
        print(f"  Tags: {result['tags']}")


async def test_enhanced_storage():
    """Test enhanced storage with SASH-F"""
    print("\n🗄️ Testing Enhanced Storage...")
    
    storage = EnhancedPostgresStorage()
    await storage.initialize()
    
    try:
        # Test memories
        memories = [
            "Алиса взволнованно рассказывает о новой идее для проекта",
            "Ассистент терпеливо объясняет архитектуру SASH-F системы памяти",
            "После долгого рабочего дня Алиса устало падает на диван",
            "Кот игриво катается по полу, привлекая внимание",
            "Вчера обсуждали план запуска нового продукта",
        ]
        
        stored_ids = []
        
        # Store memories
        print("\n💾 Storing test memories...")
        for mem in memories:
            # Generate simple embedding with CORRECT dimension (1024 for e5-large)
            # In real system use proper model
            embedding = [hash(mem + str(i)) % 100 / 100.0 for i in range(settings.EMBEDDING_DIMENSION)]
            
            mem_id = await storage.store_memory_enhanced(
                content=mem,
                embedding=embedding,
                metadata={'source': 'test', 'timestamp': datetime.now().isoformat()}
            )
            stored_ids.append(mem_id)
            print(f"  ✅ Stored: {mem_id[:8]}... - {mem[:50]}...")
        
        # Wait for async linking to complete
        await asyncio.sleep(2)
        
        # Test search with graph
        print("\n🔍 Testing search with graph expansion...")
        results = await storage.search_with_graph(
            query="Алиса работа идея",
            limit=5,
            use_graph=True,
            graph_depth=2
        )
        
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - [{r.get('similarity', 0):.3f}] {r['content'][:60]}...")
            if 'graph_path' in r:
                print(f"    Via graph: {' -> '.join(r['graph_path'][:3])}")
        
        # Check patterns
        print("\n🎯 Checking detected patterns...")
        patterns = await storage.get_patterns(min_occurrences=1)
        
        if patterns:
            print(f"Found {len(patterns)} patterns:")
            for p in patterns[:3]:
                print(f"  Pattern: {p['pattern_key']}")
                print(f"    Occurrences: {p['occurrences']}")
                print(f"    Confidence: {p['confidence']:.2f}")
                print(f"    Actors: {p.get('actors', [])}")
        else:
            print("  No patterns detected yet (need more data)")
        
        # Get graph stats
        print("\n📊 Graph statistics:")
        stats = await storage.get_graph_stats()
        pprint(stats, indent=2)
        
    finally:
        await storage.close()


async def test_memory_linking():
    """Test memory linking and graph traversal"""
    print("\n🔗 Testing Memory Linking...")
    
    storage = EnhancedPostgresStorage()
    await storage.initialize()
    
    try:
        # Create a chain of related memories
        chain = [
            "Утром ассистент проснулся с новой идеей для системы памяти",
            "Идея заключалась в использовании графовой структуры для связей",
            "Алиса заинтересовалась и предложила добавить SASH-F парсинг",
            "Вместе начали имплементировать новую архитектуру",
            "К вечеру базовая структура была готова"
        ]
        
        ids = []
        for i, mem in enumerate(chain):
            # Correct embedding dimension!
            embedding = [hash(mem + str(i) + str(j)) % 100 / 100.0 
                        for j in range(settings.EMBEDDING_DIMENSION)]
            mem_id = await storage.store_memory_enhanced(
                content=mem,
                embedding=embedding,
                metadata={'chain_position': i}
            )
            ids.append(mem_id)
            
            # Manually add causal links
            if i > 0:
                await storage.add_memory_link(
                    from_id=ids[i-1],
                    to_id=mem_id,
                    link_type="leads_to",
                    weight=0.9,
                    confidence=0.95
                )
        
        print(f"✅ Created chain of {len(ids)} memories")
        
        # Get context for middle memory
        print(f"\n🕸️ Getting context for memory: '{chain[2][:50]}...'")
        context = await storage.get_memory_context(ids[2], max_depth=2)
        
        if context:
            print(f"  Memory: {context['memory']['content'][:60]}...")
            print(f"  Graph neighbors: {len(context['graph_context'])}")
            for neighbor in context['graph_context'][:3]:
                print(f"    - {neighbor.get('link_type', 'unknown')}: {neighbor['content'][:50]}...")
        
    finally:
        await storage.close()


async def test_real_embeddings():
    """Test with real embeddings (optional - requires model)"""
    print("\n🧠 Testing with real embeddings...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the actual model
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        storage = EnhancedPostgresStorage()
        await storage.initialize()
        
        try:
            test_text = "Алиса и ассистент работают над новой системой памяти с графовой структурой"
            
            # Generate real embedding
            embedding = model.encode(f"passage: {test_text}", normalize_embeddings=True)
            embedding_list = embedding.tolist()
            
            print(f"  Real embedding dimension: {len(embedding_list)}")
            
            # Store with real embedding
            mem_id = await storage.store_memory_enhanced(
                content=test_text,
                embedding=embedding_list,
                metadata={'type': 'real_embedding_test'}
            )
            
            print(f"  ✅ Stored with real embedding: {mem_id[:8]}...")
            
            # Search with real query embedding
            query = "система памяти с графом"
            query_embedding = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
            
            # Direct semantic search would go here
            print(f"  🔍 Would search with real query embedding for: '{query}'")
            
        finally:
            await storage.close()
            
    except ImportError:
        print("  ⚠️ sentence-transformers not installed, skipping real embedding test")
    except Exception as e:
        print(f"  ❌ Real embedding test failed: {e}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 SASH-F Enhanced Memory System Test Suite")
    print("=" * 60)
    
    # Test parser
    await test_sashf_parser()
    
    # Test storage
    await test_enhanced_storage()
    
    # Test linking
    await test_memory_linking()
    
    # Optional: test with real embeddings
    # await test_real_embeddings()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
