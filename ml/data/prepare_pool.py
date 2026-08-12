import json
import os
import random
import hashlib

from ml.data.dedupe import normalize, SeenSet

def main():
    random.seed(42)

    sources = [
        "ml/data/generated/router.jsonl",
        "ml/data/generated/finance.jsonl",
        "ml/data/generated/journal.jsonl"
    ]

    pool = []
    seen = SeenSet()

    stats = {}
    dropped = 0

    for source in sources:
        if not os.path.exists(source):
            continue

        source_name = os.path.basename(source).split('.')[0]
        source_count = 0

        with open(source, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if not text:
                    continue

                if not seen.add_and_check(text):
                    dropped += 1
                    continue

                cid = hashlib.sha256(text.encode('utf-8')).hexdigest()[:10]

                pool.append({
                    "id": cid,
                    "text": text,
                    "source": f"generated:{source_name}"
                })
                source_count += 1

        stats[source_name] = source_count

    random.shuffle(pool)

    os.makedirs("ml/data/candidates", exist_ok=True)
    pool_path = "ml/data/candidates/pool.jsonl"
    with open(pool_path, 'w', encoding='utf-8') as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")

    print(f"Created pool at {pool_path}")
    for src, count in stats.items():
        print(f"  {src}: {count} candidates")
    print(f"  Dropped as cross-source duplicates: {dropped}")
    print(f"  Final pool size: {len(pool)}")

if __name__ == "__main__":
    main()
