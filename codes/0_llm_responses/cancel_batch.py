from openai import OpenAI

client = OpenAI()


# List batches (you can increase limit if you want more than 10)
batches = client.batches.list(limit=20)

print(f"{'ID':<40} {'STATUS':<15} {'CREATED_AT'}")
print("-" * 70)
for b in batches.data:
    print(f"{b.id:<40} {b.status:<15} {b.created_at}")
    

# client.batches.cancel("batch_68de2eabf4a481908fa669d1d7a8c7e4")

# with client.chat.completions.stream(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Write me a poem"}],
# ) as stream:
#     for event in stream:
#         if event.type == "token":
#             print(event.token.text, end="", flush=True)  # token by token
#     print("\n---\nDone.")

# a = client.batches.retrieve('batch_68dfe9aacab08190b1a827faf6125e34')
# print(a)
# client.batches.retrieve('batch/idcg4e3rfm7otl8xheelsc3x9hdau7yqu2unj7')   

from google import genai
client = genai.Client()

for batch in client.batches.list():
    print(batch.name, batch.state)