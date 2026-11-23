#!/usr/bin/env python3
"""Re-chunk existing long memories that were stored before chunking was implemented.

Usage:
    python -m scripts.rechunk_existing --dry-run   # preview what would be chunked
    python -m scripts.rechunk_existing              # actually do it
"""
import asyncio
import argparse
import sys
import os
import hashlib
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from config.settings import settings
from src.core.chunker import chunk_text, needs_chunking


async def init_conn(conn):
    await register_vector(conn)


async def main():
    parser = argparse.ArgumentParser(description="Re-chunk existing long memories")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of memories to process (0=all)")
    args = parser.parse_args()

    pool = await asyncpg.create_pool(settings.PG_DSN, min_size=2, max_size=5, init=init_conn)

    # Load encoder lazily
    encoder = None

    async def encode_text(text: str) -> np.ndarray:
        nonlocal encoder
        if encoder is None:
            print("Loading BGE-M3 model...", file=sys.stderr)
            from src.core.encoder import load_bge_m3
            encoder = load_bge_m3(settings.EMBEDDING_MODEL, use_fp16=True)
            print("Model loaded.", file=sys.stderr)

        result = encoder.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result['dense_vecs'][0]

    async with pool.acquire() as conn:
        # Find long memories without existing chunks
        query = """
            SELECT m.id, m.content, LENGTH(m.content) as content_length
            FROM memories m
            WHERE m.parent_id IS NULL
            AND LENGTH(m.content) > $1
            AND NOT EXISTS (
                SELECT 1 FROM memories c WHERE c.parent_id = m.id
            )
            ORDER BY m.created_at DESC
        """
        params = [settings.CHUNK_MAX_SIZE]

        if args.limit > 0:
            query += " LIMIT $2"
            params.append(args.limit)

        candidates = await conn.fetch(query, *params)

    print(f"\nFound {len(candidates)} memories eligible for chunking", file=sys.stderr)

    if not candidates:
        print("Nothing to do.", file=sys.stderr)
        return

    # Stats
    total_chunks = 0
    processed = 0
    skipped = 0

    for row in candidates:
        memory_id = row['id']
        content = row['content']
        content_len = row['content_length']

        chunks = chunk_text(content)
        if not chunks:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [{memory_id}] {content_len} chars -> {len(chunks)} chunks")
            for c in chunks:
                print(f"    chunk {c.index}: {len(c.text)} chars | {c.text[:80]}...")
            total_chunks += len(chunks)
            processed += 1
            continue

        # Actually create chunks
        async with pool.acquire() as conn:
            async with conn.transaction():
                for chunk in chunks:
                    chunk_embedding = await encode_text(chunk.text)
                    chunk_embedding_array = chunk_embedding.astype(np.float32)
                    chunk_id = f"{memory_id}_c{chunk.index}"

                    await conn.execute("""
                        INSERT INTO memories (id, content, embedding, metadata, sashf, parent_id, chunk_index, created_at, updated_at)
                        SELECT $1, $2, $3, metadata, '{}'::jsonb, $4, $5, created_at, NOW()
                        FROM memories WHERE id = $4
                        ON CONFLICT (id) DO NOTHING
                    """, chunk_id, chunk.text, chunk_embedding_array, memory_id, chunk.index)

                # Null out parent's embedding (search should go through chunks now)
                await conn.execute("""
                    UPDATE memories SET embedding = NULL, updated_at = NOW()
                    WHERE id = $1
                """, memory_id)

        total_chunks += len(chunks)
        processed += 1
        print(f"  [{memory_id}] {content_len} chars -> {len(chunks)} chunks", file=sys.stderr)

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary:", file=sys.stderr)
    print(f"  Processed: {processed}", file=sys.stderr)
    print(f"  Skipped (too short after analysis): {skipped}", file=sys.stderr)
    print(f"  Total chunks created: {total_chunks}", file=sys.stderr)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
