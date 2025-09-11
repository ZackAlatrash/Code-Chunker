import json
import argparse
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--index", default="code-chunks")
    ap.add_argument("--jsonl", default="chunks.jsonl")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    client = OpenSearch(args.host)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # sanity: model must match mapping dimension
    dim = model.get_sentence_embedding_dimension()
    if dim != 384:
        raise RuntimeError(f"Index mapping is 384D but model returned {dim}D")

    def stream():
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    buf = []
    def bulk_actions(batch):
        # You can optionally prepend small context like the file path:
        texts = [f"[{d.get('path')}] \n{d['text']}" for d in batch]
        # normalize_embeddings=True is important for cosinesimil
        vecs = model.encode(texts, normalize_embeddings=True).tolist()
        for d, v in zip(batch, vecs):
            src = dict(d)
            src["embedding"] = v
            yield {
                "_op_type": "index",
                "_index": args.index,
                "_id": src["id"],
                "_source": src,
            }

    total_ok = 0
    for d in stream():
        buf.append(d)
        if len(buf) >= args.batch:
            ok, errors = helpers.bulk(
                client, bulk_actions(buf),
                request_timeout=300, raise_on_error=False
            )
            total_ok += ok
            buf.clear()
            if errors:
                print(f"[WARN] Some items failed in last batch (showing one): {errors[:1]}")

    if buf:
        ok, errors = helpers.bulk(
            client, bulk_actions(buf),
            request_timeout=300, raise_on_error=False
        )
        total_ok += ok
        if errors:
            print(f"[WARN] Some items failed in last batch (showing one): {errors[:1]}")

    # OpenSearch client v2+ uses keyword-only params for indices APIs
    client.indices.refresh(index=args.index)
    count = client.count(index=args.index)["count"]
    print(f"Ingested OK: {total_ok} | Index doc count: {count}")

if __name__ == "__main__":
    main()
