from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print("Kelsie AI ready — type 'quit' to exit")

history = []

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        print("Goodbye.")
        break

    history.append(f"User: {prompt}")
    input_text = " ".join(history[-6:])  # last 6 turns
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    response = ""
    while not response.strip():  # retry until non-empty
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8
            )
        response = tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )

    print(f"Kelsie: {response}")
    history.append(f"Kelsie: {response}")

