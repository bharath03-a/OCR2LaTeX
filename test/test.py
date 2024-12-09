import ollama

try:
    response = ollama.chat(
        model="llama3.2-vision",
        messages=[{"role": "user", "content": "Generate LaTeX for the formula x^2 + y^2 = z^2"}],
    )
    print(response)
except Exception as e:
    print(f"Error: {e}")