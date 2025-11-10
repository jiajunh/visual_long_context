import os
import json

from tqdm import tqdm

from openai import OpenAI

OPENAI_API_KEY=""


def generate_paragraphs(prompt, image_url, model="gpt-5-nano"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                ],
            }
        ],
    )
    return response.output_text


if __name__ == "__main__":
    original_data_path = "./data/samples_500/captions.json"
    output_data_path = "./data/samples_500/agent_conversation.json"

    model = "gpt-5"

    prompt = """Construct three long conversations start with the image. 
    Each conversation should contain about 10 paragraphs, and each paragraph is about 100 words. 
    Do not just limit the content to describing the image details. 
    You may dive deeply into a specific aspect or transition into different topics as the conversation goes.
    The first conversation should stay most focused on the image. 
    The second conversation should be more diverse and may go beyond the image. 
    The third conversation should be the most creative and diverse. It can deviate heavily from the image and jump to other topics.
    Return the response in a json format. The keys are 'difficulty_1', 'difficulty_2' and 'difficulty_3' 
    Each value is a list of the paragraphs."""

    with open(original_data_path) as f:
        data = json.load(f)

    data_with_conversation = {}

    for key, val in tqdm(data.items(), desc="Generating Conversation"):
        response = generate_paragraphs(prompt=prompt, image_url=val["url"], model="gpt-5")
        response_dict = json.loads(response)
        data_with_conversation[key] = {**val, **response_dict}
        break

    with open(output_data_path, "a", encoding="utf-8") as f:
        json.dump(data_with_conversation, f, ensure_ascii=False)