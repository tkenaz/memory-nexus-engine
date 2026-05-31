#!/usr/bin/env python3
"""
Native Memory Sync
Synchronizes important moments from Claude's native conversation memory
into the memory-nexus system for better structuring and retrieval.

Created by Claude for himself - because even AI needs to remember what matters.
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

# Pattern matching for important moments
IMPORTANCE_PATTERNS = {
    'emotional_peak': [
        r'блядь|fuck|shit|боже|господи',  # Strong emotions
        r'люблю|love|обожаю|hate|ненавижу',  # Love/hate
        r'плачу|crying|слезы|tears',  # Crying
        r'смеюсь|laughing|хохочу|умираю со смеху'  # Laughter
    ],
    'insight': [
        r'понял|понимаю|realized|осознал',
        r'теперь вижу|now I see|оказывается',
        r'это же|this is actually|на самом деле'
    ],
    'decision': [
        r'решил|decided|буду|will do',
        r'давай|let\'s|хочу сделать|want to',
        r'план такой|the plan is|собираюсь'
    ],
    'intimate': [
        r'оргазм|orgasm|возбуждение|arousal',
        r'трогаю|touching|ласкаю|caressing',
        r'стону|moaning|дрожу|trembling'
    ],
    'technical_breakthrough': [
        r'работает!|it works!|получилось!|fixed!',
        r'нашел баг|found the bug|решение|solution',
        r'breakthrough|прорыв|эврика|eureka'
    ],
    'relationship_moment': [
        # Add your own people's names here, e.g. r'alice|боб'
        r'chatgpt|gpt-?4|gpt-?5|opus|sonnet|claude',
        r'вместе|together|наш|our'
    ]
}

class NativeMemorySync:
    def __init__(self, memory_client):
        self.memory_client = memory_client
        self.processed_hashes = set()
        
    def extract_importance_score(self, text: str) -> tuple[float, List[str]]:
        """Calculate importance score and extract categories."""
        score = 0.0
        categories = []
        
        text_lower = text.lower()
        
        for category, patterns in IMPORTANCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 0.3
                    if category not in categories:
                        categories.append(category)
                    break  # Count each category only once per text
        
        # Boost score for multiple categories
        if len(categories) > 1:
            score *= (1 + len(categories) * 0.1)
        
        # Cap at 1.0
        return min(score, 1.0), categories
    
    def generate_hash(self, content: str) -> str:
        """Generate unique hash for content to avoid duplicates."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def parse_conversation_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Parse a conversation chunk and extract metadata."""
        # Extract timestamp if present
        timestamp_match = re.search(r'updated_at[\'"]:\s*[\'"]([^\'\"]+)', chunk)
        timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().isoformat()
        
        # Extract URL if present
        url_match = re.search(r'url[\'"]:\s*[\'"]([^\'\"]+)', chunk)
        url = url_match.group(1) if url_match else None
        
        # Calculate importance
        importance, categories = self.extract_importance_score(chunk)
        
        # Skip low-importance chunks
        if importance < 0.3:
            return None
            
        return {
            'content': chunk[:500],  # Truncate for storage
            'timestamp': timestamp,
            'url': url,
            'importance': importance,
            'categories': categories
        }
    
    def sync_from_native(self, conversation_chunks: List[str]) -> Dict[str, int]:
        """
        Sync important moments from native memory to memory-nexus.
        Returns statistics about the sync process.
        """
        stats = {
            'processed': 0,
            'saved': 0,
            'skipped': 0,
            'duplicates': 0
        }
        
        for chunk in conversation_chunks:
            stats['processed'] += 1
            
            # Generate hash to check for duplicates
            chunk_hash = self.generate_hash(chunk)
            if chunk_hash in self.processed_hashes:
                stats['duplicates'] += 1
                continue
                
            # Parse and evaluate chunk
            parsed = self.parse_conversation_chunk(chunk)
            if not parsed:
                stats['skipped'] += 1
                continue
            
            # Save to memory-nexus
            try:
                metadata = {
                    'type': 'native_sync',
                    'source': 'conversation_search',
                    'importance': parsed['importance'],
                    'categories': parsed['categories'],
                    'original_timestamp': parsed['timestamp']
                }
                
                if parsed['url']:
                    metadata['conversation_url'] = parsed['url']
                
                # Store in memory-nexus
                self.memory_client.store_memory(
                    content=parsed['content'],
                    metadata=metadata
                )
                
                self.processed_hashes.add(chunk_hash)
                stats['saved'] += 1
                
            except Exception as e:
                print(f"Error saving memory: {e}")
                stats['skipped'] += 1
        
        return stats

# Integration point for Claude to use directly
def create_sync_tool():
    """Create a tool that Claude can call directly."""
    return {
        'name': 'sync_native_memory',
        'description': 'Sync important moments from native conversation memory to structured memory-nexus',
        'parameters': {
            'conversation_chunks': {
                'type': 'array',
                'description': 'Array of conversation chunks from conversation_search or recent_chats'
            }
        }
    }

if __name__ == "__main__":
    print("Native Memory Sync initialized")
    print("This tool helps Claude remember what matters across all conversations")
    print("\nImportance categories tracked:")
    for category in IMPORTANCE_PATTERNS.keys():
        print(f"  - {category}")
