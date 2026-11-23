"""Build/refresh the tag_embeddings catalog from memories.metadata->tags.

Collects the tag vocabulary (normalised with lower()), embeds missing tags and
refreshes memory_count. New tags are also added incrementally when memories are
stored; run this script after a bulk import, and with --rebuild after
reembed_all.py whenever the embedding model changes.

Usage:
    python build_tag_index.py [--rebuild] [--batch-size 128]
        --rebuild  re-embed ALL tags (required after changing the embedding model)
"""
import argparse
import asyncio
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.postgres_enhanced import EnhancedPostgresStorage


async def build(rebuild: bool = False, batch_size: int = 128):
    storage = EnhancedPostgresStorage()
    await storage.initialize()

    print("Loading embedding model...")
    await storage._ensure_encoder()
    print("Model loaded.\n")

    async with storage.pool.acquire() as conn:
        # Tag vocabulary with frequencies (parents/standalone only: chunks duplicate metadata)
        vocab = await conn.fetch("""
            SELECT lower(t.tag) AS tag, count(*) AS n
            FROM memories m
            JOIN LATERAL jsonb_array_elements_text(
                CASE WHEN jsonb_typeof(m.metadata->'tags') = 'array'
                     THEN m.metadata->'tags' ELSE '[]'::jsonb END
            ) AS t(tag) ON TRUE
            WHERE m.parent_id IS NULL AND length(trim(t.tag)) > 0
            GROUP BY lower(t.tag)
        """)
        counts = {r['tag']: r['n'] for r in vocab}
        print(f"Tag vocabulary: {len(counts)} distinct tags")

        existing = {r['tag'] for r in await conn.fetch("SELECT tag FROM tag_embeddings")}
        to_embed = sorted(counts) if rebuild else sorted(set(counts) - existing)
        print(f"To embed: {len(to_embed)} ({'rebuild' if rebuild else 'missing only'})")

        start = time.time()
        done = 0
        loop = asyncio.get_event_loop()
        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i:i + batch_size]
            result = await loop.run_in_executor(
                None,
                lambda b=batch: storage.encoder.encode(
                    b, return_dense=True, return_sparse=False, return_colbert_vecs=False
                )
            )
            for tag, emb in zip(batch, result['dense_vecs']):
                await conn.execute("""
                    INSERT INTO tag_embeddings (tag, embedding, memory_count, updated_at)
                    VALUES ($1, $2, $3, now())
                    ON CONFLICT (tag) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        memory_count = EXCLUDED.memory_count,
                        updated_at = now()
                """, tag, np.array(emb, dtype=np.float32), counts[tag])
                done += 1
            rate = done / (time.time() - start) if done else 0
            print(f"  [{done}/{len(to_embed)}] {rate:.0f} tags/s", end='\r')

        # Refresh counters for tags that were not re-embedded
        stale = set(counts) & existing if not rebuild else set()
        for tag in stale:
            await conn.execute(
                "UPDATE tag_embeddings SET memory_count = $2, updated_at = now() WHERE tag = $1",
                tag, counts[tag])

        # Remove tags that left the vocabulary (memory deleted or re-tagged)
        removed = await conn.execute(
            "DELETE FROM tag_embeddings WHERE NOT (tag = ANY($1))", list(counts))

        print(f"\nDone: embedded {done}, counts refreshed {len(stale)}, "
              f"orphans removed: {removed.split()[-1]}, total vocab {len(counts)}")

    await storage.close()


def main():
    parser = argparse.ArgumentParser(description="Build tag embedding catalog")
    parser.add_argument("--rebuild", action="store_true", help="Re-embed all tags")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    asyncio.run(build(rebuild=args.rebuild, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
