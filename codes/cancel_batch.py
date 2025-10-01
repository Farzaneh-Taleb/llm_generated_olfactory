from openai import OpenAI

client = OpenAI()


# List batches (you can increase limit if you want more than 10)
batches = client.batches.list(limit=20)

print(f"{'ID':<40} {'STATUS':<15} {'CREATED_AT'}")
print("-" * 70)
for b in batches.data:
    print(f"{b.id:<40} {b.status:<15} {b.created_at}")