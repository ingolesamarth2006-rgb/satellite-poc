from core.semantic_search import semantic_search


query = "a river surrounded by vegetation"

results = semantic_search(query, top_k=5)

print("\n========== TOP RESULTS ==========")

for i, result in enumerate(results, start=1):
    print(f"\n{i}. Category: {result['category']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Path: {result['path']}")

print("\n=================================")