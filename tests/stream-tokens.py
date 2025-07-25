from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = "Aclevo/Lazarus-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create an iterator streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

while True:
	ui = input("ask lazarus a question: ")
	prompt = "You are an intelligent AI assistant named Lazarus, who will answer user questions to the best of Lazarus's ability. " + ui
	inputs = tokenizer(prompt, return_tensors="pt")

# Run generation in a separate thread
	generation_kwargs = dict(
    		**inputs,
    		max_new_tokens=100,
    		streamer=streamer,
    		do_sample=True,
    		temperature=0.7
	)

	thread = Thread(target=model.generate, kwargs=generation_kwargs)
	thread.start()

# Process tokens as they come
	for new_text in streamer:
    		print(new_text, end="", flush=True)
    # You can do custom processing here
