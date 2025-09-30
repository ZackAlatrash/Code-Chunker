# bulk_index_chunks.py
# pip install opensearch-py sentence-transformers tqdm
import json, sys, os, math
from tqdm import tqdm
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

INDEX = "code_chunks_v2"   # created earlier
BATCH = 500                # bulk size

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main(jsonl_path, host="http://localhost:9200"):
    client = OpenSearch(hosts=[host])
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # dim=384

    def gen_actions():
        for doc in read_jsonl(jsonl_path):
            vec = model.encode(doc["text"], normalize_embeddings=True).tolist()
            doc["vector"] = vec
            yield {
                "_op_type": "index",
                "_index": INDEX,
                "_id": doc["id"],   # stable id path#chunk
                "_source": doc
            }

    # stream in batches
    success, fail = helpers.bulk(client, gen_actions(), chunk_size=BATCH, request_timeout=120)
    print(f"Indexed: {success}, Failed: {fail}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bulk_index_chunks.py chunks.jsonl [http://localhost:9200]")
        sys.exit(1)
    host = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9200"
    main(sys.argv[1], host)