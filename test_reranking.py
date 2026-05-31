#!/usr/bin/env python3
"""Тест для проверки cross-encoder reranking"""
import asyncio
import sys
from datetime import datetime

from src.search.hybrid_search import HybridSearch
from src.models.memory_simple import SearchQuery


async def test_reranking(query_text: str):
    """Сравнение результатов с и без reranking"""

    print(f"\n{'='*80}")
    print(f"ТЕСТ RERANKING")
    print(f"{'='*80}")
    print(f"Query: {query_text}")
    print(f"{'='*80}\n")

    search = HybridSearch()
    await search.initialize()

    try:
        # 1. Поиск БЕЗ reranking
        print("🔍 Поиск БЕЗ reranking...")
        query_no_rerank = SearchQuery(
            query=query_text,
            limit=20,  # Берем больше чтобы было что rerank'ать
            use_hybrid=True,
            use_reranking=False
        )
        results_no_rerank = await search.search(query_no_rerank)

        print(f"\nНайдено: {len(results_no_rerank)} результатов\n")
        print("TOP 10 БЕЗ RERANKING:")
        print("-" * 80)
        for i, res in enumerate(results_no_rerank[:10], 1):
            content_preview = res.memory.content[:100].replace('\n', ' ')
            print(f"{i}. [Score: {res.score:.4f}] [{res.search_type}]")
            print(f"   {content_preview}...")
            print(f"   Created: {res.memory.created_at.strftime('%Y-%m-%d %H:%M')}")
            print()

        # 2. Поиск С reranking
        print("\n" + "="*80)
        print("🎯 Поиск С RERANKING...")
        query_with_rerank = SearchQuery(
            query=query_text,
            limit=20,
            use_hybrid=True,
            use_reranking=True
        )
        results_with_rerank = await search.search(query_with_rerank)

        print(f"\nНайдено: {len(results_with_rerank)} результатов\n")
        print("TOP 10 С RERANKING:")
        print("-" * 80)
        for i, res in enumerate(results_with_rerank[:10], 1):
            content_preview = res.memory.content[:100].replace('\n', ' ')
            print(f"{i}. [Score: {res.score:.4f}] [{res.search_type}]")
            print(f"   {content_preview}...")
            print(f"   Created: {res.memory.created_at.strftime('%Y-%m-%d %H:%M')}")
            print()

        # 3. Сравнение
        print("\n" + "="*80)
        print("📊 СРАВНЕНИЕ:")
        print("-" * 80)

        # Получаем ID результатов
        ids_no_rerank = [r.memory.id for r in results_no_rerank[:10]]
        ids_with_rerank = [r.memory.id for r in results_with_rerank[:10]]

        # Новые в топ-10 после reranking
        new_in_top = set(ids_with_rerank) - set(ids_no_rerank)
        if new_in_top:
            print(f"\n✨ Новые результаты в топ-10 после reranking: {len(new_in_top)}")
            for mem_id in new_in_top:
                res = next(r for r in results_with_rerank if r.memory.id == mem_id)
                preview = res.memory.content[:80].replace('\n', ' ')
                print(f"   - {preview}...")

        # Ушли из топ-10
        removed_from_top = set(ids_no_rerank) - set(ids_with_rerank)
        if removed_from_top:
            print(f"\n❌ Убрано из топ-10 после reranking: {len(removed_from_top)}")

        # Изменение порядка
        same_ids = set(ids_no_rerank) & set(ids_with_rerank)
        if same_ids:
            print(f"\n🔄 Общих результатов: {len(same_ids)}")

            # Показываем насколько изменились позиции
            position_changes = []
            for mem_id in same_ids:
                old_pos = ids_no_rerank.index(mem_id) + 1
                new_pos = ids_with_rerank.index(mem_id) + 1
                change = old_pos - new_pos
                if change != 0:
                    position_changes.append((mem_id, old_pos, new_pos, change))

            if position_changes:
                position_changes.sort(key=lambda x: abs(x[3]), reverse=True)
                print("\n   Наибольшие изменения позиций:")
                for mem_id, old_pos, new_pos, change in position_changes[:5]:
                    direction = "⬆️" if change > 0 else "⬇️"
                    res = next(r for r in results_with_rerank if r.memory.id == mem_id)
                    preview = res.memory.content[:60].replace('\n', ' ')
                    print(f"   {direction} {old_pos} → {new_pos} (Δ{change:+d}): {preview}...")

        # Score distribution
        print("\n📈 Score distribution:")
        if results_no_rerank:
            print(f"   БЕЗ rerank: max={max(r.score for r in results_no_rerank[:10]):.4f}, "
                  f"min={min(r.score for r in results_no_rerank[:10]):.4f}")
        if results_with_rerank:
            print(f"   С rerank:   max={max(r.score for r in results_with_rerank[:10]):.4f}, "
                  f"min={min(r.score for r in results_with_rerank[:10]):.4f}")

    finally:
        await search.close()

    print("\n" + "="*80)
    print("✅ Тест завершен")
    print("="*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_reranking.py 'your search query'")
        print("\nExamples:")
        print("  python test_reranking.py 'что я думала про семантический поиск'")
        print("  python test_reranking.py 'october'")
        print("  python test_reranking.py 'project ideas'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    asyncio.run(test_reranking(query))
