import sys
sys.path.insert(0, '.')

# Testing the chunking process
from campus_policy_rag import chunk_text

# Test 1: Basic chunking
test_text = "This is a sample document about campus policies. Students must follow all rules and regulations set by the campus administration. The library has specific hours of operation. Hostel facilities are available for eligible students. Refund policies apply to all paid services."

print("=" * 60)
print("TESTING CHUNKING PROCESS")
print("=" * 60)

# Test with different parameters
chunks_small = chunk_text(test_text, chunk_size=10, overlap_words=2)
print(f"\nTest 1: Chunking with chunk_size=10, overlap_words=2")
print(f"Total chunks created: {len(chunks_small)}")
for idx, chunk in enumerate(chunks_small):
    print(f"Chunk {idx}: ({len(chunk.split())} words) {chunk[:60]}...")

chunks_medium = chunk_text(test_text, chunk_size=20, overlap_words=5)
print(f"\nTest 2: Chunking with chunk_size=20, overlap_words=5")
print(f"Total chunks created: {len(chunks_medium)}")
for idx, chunk in enumerate(chunks_medium):
    print(f"Chunk {idx}: ({len(chunk.split())} words) {chunk[:60]}...")

chunks_large = chunk_text(test_text, chunk_size=120, overlap_words=30)
print(f"\nTest 3: Chunking with chunk_size=120, overlap_words=30 (default)")
print(f"Total chunks created: {len(chunks_large)}")
for idx, chunk in enumerate(chunks_large):
    print(f"Chunk {idx}: ({len(chunk.split())} words) {chunk[:60]}...")

# Test edge cases
empty_text = ""
empty_chunks = chunk_text(empty_text)
print(f"\nTest 4: Empty text")
print(f"Total chunks: {len(empty_chunks)} (expected: 0)")

small_text = "Only a few words"
small_chunks = chunk_text(small_text)
print(f"\nTest 5: Text smaller than chunk_size")
print(f"Total chunks: {len(small_chunks)}")
print(f"Chunk: {small_chunks[0] if small_chunks else 'None'}")

print("\n" + "=" * 60)
print("CHUNKING VERIFICATION COMPLETE")
print("=" * 60)
