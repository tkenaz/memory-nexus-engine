"""Re-embed all memories with the current BGE-M3 model.

Fixes the model mismatch issue where old memories were encoded with a different
embedding model. Processes all memories (standalone + chunks) in batches.

Usage:
    python reembed_all.py [--batch-size 32] [--dry-run]
"""
import asyncio
import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.postgres_enhanced import EnhancedPostgresStorage
from config.settings import settings


async def reembed_all(batch_size: int = 32, dry_run: bool = False):
    storage = EnhancedPostgresStorage()
    await storage.initialize()

    # Ensure model is loaded before we start timing
    print("Loading BGE-M3 model...")
    await storage._ensure_encoder()
    print("Model loaded.\n")

    async with storage.pool.acquire() as conn:
        # Get all memories (both standalone and chunks)
        rows = await conn.fetch("""
            SELECT id, content, parent_id, chunk_index
            FROM memories
            ORDER BY created_at ASC
        """)

        total = len(rows)
        print(f"Total memories to re-embed: {total}")

        if dry_run:
            print("DRY RUN — no changes will be made.")
            return

        updated = 0
        skipped = 0
        errors = 0
        start = time.time()

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            texts = []
            ids = []

            for r in batch:
                # For chunks, embed chunk text; for standalone/parents, embed full content
                text = r['content']
                if not text or not text.strip():
                    skipped += 1
                    continue
                # Truncate very long texts for embedding (BGE-M3 max ~8192 tokens)
                texts.append(text[:8000])
                ids.append(r['id'])

            if not texts:
                continue

            try:
                # Batch encode
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda t=texts: storage.encoder.encode(
                        t,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False
                    )
                )
                embeddings = result['dense_vecs']

                # Update in DB
                for j, (mem_id, emb) in enumerate(zip(ids, embeddings)):
                    emb_array = np.array(emb, dtype=np.float32)
                    await conn.execute(
                        "UPDATE memories SET embedding = $1 WHERE id = $2",
                        emb_array, mem_id
                    )
                    updated += 1

            except Exception as e:
                print(f"  ERROR batch {i}: {e}")
                errors += len(texts)
                continue

            elapsed = time.time() - start
            rate = updated / elapsed if elapsed > 0 else 0
            eta = (total - updated) / rate if rate > 0 else 0
            print(f"  [{updated}/{total}] {rate:.1f}/s, ETA {eta:.0f}s", end='\r')

        elapsed = time.time() - start
        print(f"\n\nDone! Updated: {updated}, Skipped: {skipped}, Errors: {errors}")
        print(f"Time: {elapsed:.1f}s ({updated/elapsed:.1f} memories/s)")


def main():
    parser = argparse.ArgumentParser(description="Re-embed all memories with current model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--dry-run", action="store_true", help="Don't update, just count")
    args = parser.parse_args()

    asyncio.run(reembed_all(batch_size=args.batch_size, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
