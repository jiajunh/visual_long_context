import os
import json

from tqdm import tqdm

from openai import OpenAI

OPENAI_API_KEY=""

def generate_paragraphs(api_key, prompt, model="gpt-5-nano"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=model,
        reasoning={"effort": "minimal"},
        input=prompt,
    )
    return response.output_text


if __name__ == "__main__":
    output_path = "./data/random_paragraphs.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_paragraphs = 1000

    prompt = "Provide one paragraph on a random topic, about 100 words."

    with open(output_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(total_paragraphs), desc="Generating"):
            response = generate_paragraphs(api_key=OPENAI_API_KEY, prompt=prompt)
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
            f.flush()
    