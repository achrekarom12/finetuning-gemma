import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText

# 1. Setup Device and Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device)

def chat_with_gemma():
    # Initialize chat history
    messages = []
    print("--- Chat started (Type 'exit' to stop) ---")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        image_path = input("Image Path/URL (leave blank if none): ").strip()
        
        # Prepare the message content
        content = [{"type": "text", "text": user_input}]
        
        raw_image = None
        if image_path:
            try:
                if image_path.startswith("http"):
                    raw_image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    raw_image = Image.open(image_path)
                content.append({"type": "image"})
            except Exception as e:
                print(f"Error loading image: {e}")

        # Add to history
        messages.append({"role": "user", "content": content})

        # 2. Process Input
        # Note: apply_chat_template handles the formatting for Gemma-4
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(
            text=prompt, 
            images=raw_image, 
            return_tensors="pt"
        ).to(device)

        # 3. Generate Response
        output = model.generate(**inputs, max_new_tokens=512)
        
        # Decode only the newly generated tokens
        generated_text = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]

        print(f"\nAssistant: {generated_text}\n")
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": [{"type": "text", "text": generated_text}]})

if __name__ == "__main__":
    chat_with_gemma()