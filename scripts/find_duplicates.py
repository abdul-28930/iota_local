"""
Find documents that are near-duplicates of each other (same underlying PDF,
different filename) already sitting in the database, and optionally delete
the extras -- keeping one canonical copy per real document.

Since content_hash wasn't tracked before this fix, existing duplicates can't
be matched by hash. Instead, this groups documents by (total_pages, first
150 chars of their first chunk's content) as a practical similarity signature
-- exact hash matching isn't possible retroactively without re-reading every
source PDF, but this catches the CGIoT-style case (same file, different name)
reliably since duplicate ingests produce identical page counts and identical
opening text.

Usage:
  python find_duplicates.py            # dry run, just shows what it would delete
  python find_duplicates.py --delete   # actually deletes the extras
"""
import os, sys, csv
from pathlib import Path
from collections import defaultdict
from supabase import create_client
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def find_duplicate_groups():
    docs = sb.table("documents_local").select("id, document_name, total_pages, created_at").execute().data
    groups = defaultdict(list)

    for doc in docs:
        # Chunks near the start are often shared generic boilerplate (cover
        # pages, disclaimers, TLP notices) that's near-identical across many
        # DIFFERENT real documents -- this caused false positives earlier
        # (e.g. flagging distinct policy templates as duplicates). Sampling a
        # chunk from ~30% into the document is far more likely to be genuinely
        # distinctive content, unique to that specific document.
        all_chunks = sb.table("sama_nora_chunks_local") \
            .select("content").eq("document_id", doc["id"]) \
            .order("id").execute().data
        if not all_chunks:
            signature_text = ""
        else:
            sample_idx = max(0, int(len(all_chunks) * 0.3))
            signature_text = all_chunks[sample_idx]["content"][:200]
        key = (doc.get("total_pages"), signature_text)
        groups[key].append(doc)

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1 and k[1]}  # ignore empty-signature group
    return duplicate_groups


def main():
    delete_mode = "--delete" in sys.argv
    groups = find_duplicate_groups()

    if not groups:
        print("No duplicate groups found.")
        return

    csv_path = Path(__file__).parent / "duplicate_report.csv"
    csv_rows = []
    total_to_delete = 0

    for key, docs in groups.items():
        docs_sorted = sorted(docs, key=lambda d: d["created_at"])  # keep the earliest ingested copy
        keep, extras = docs_sorted[0], docs_sorted[1:]
        group_id = f"{key[0]}pages_{abs(hash(key[1])) % 100000}"

        print(f"\nGroup ({len(docs)} copies, {key[0]} pages):")
        print(f"  KEEP:   {keep['document_name']} (id={keep['id']})")
        csv_rows.append({
            "group_id": group_id, "action": "KEEP", "document_id": keep["id"],
            "document_name": keep["document_name"], "total_pages": key[0],
            "created_at": keep["created_at"], "signature_preview": key[1][:80],
        })

        for e in extras:
            print(f"  DELETE: {e['document_name']} (id={e['id']})")
            total_to_delete += 1
            csv_rows.append({
                "group_id": group_id, "action": "DELETE" if delete_mode else "WOULD_DELETE",
                "document_id": e["id"], "document_name": e["document_name"], "total_pages": key[0],
                "created_at": e["created_at"], "signature_preview": key[1][:80],
            })

        if delete_mode:
            for e in extras:
                sb.table("sama_nora_chunks_local").delete().eq("document_id", e["id"]).execute()
                sb.table("documents_local").delete().eq("id", e["id"]).execute()

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "action", "document_id", "document_name",
                                                "total_pages", "created_at", "signature_preview"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'Deleted' if delete_mode else 'Would delete'} {total_to_delete} duplicate document(s) "
          f"across {len(groups)} group(s).")
    print(f"Full report written to: {csv_path}")
    if not delete_mode:
        print("Run with --delete to actually remove them.")


if __name__ == "__main__":
    main()
