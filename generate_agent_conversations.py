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
                    { "type": "input_text", "text": prompt},
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

    model = "gpt-5-nano"
    num_paragraph = 10
    paragraph_length = 100

    prompt = f"""Construct three long context start with the image, as descriptions for every paragraph.\
    Each context should contain about {num_paragraph} paragraphs, and each paragraph is about {paragraph_length} words.\
    Paragraph 1 in each context must start with something highly relevant to the image.\
    As the paragraphs progress, the topic may gradually expand to related ideas,\
    but each new paragraph should still maintain at least a loose connection to something previously mentioned,\
    even if the connection becomes weaker.\
    The first context should stay almost most focused on topic related to the image.\
    Almost every paragraph should relate to the image or its direct concepts.\
    The second context should be more diverse and may go beyond the image.\
    Later paragraphs may explore broader ideas, but must still connect to elements introduced earlier.\
    The third context should be the most creative and diverse. Paragraphs can shift topics boldly,\
    but every transition must still trace back to something mentioned before.\
    Return the response in a json format. The keys are 'difficulty_1', 'difficulty_2' and 'difficulty_3'\
    Each value is a list of the paragraphs."""

    print(prompt)

    with open(original_data_path) as f:
        data = json.load(f)

    data_with_conversation = {}

    for key, val in tqdm(data.items(), desc="Generating Conversation"):
        response = generate_paragraphs(prompt=prompt, image_url=val["url"], model="gpt-5")
        response_dict = json.loads(response)
        data_with_conversation[key] = {**val, **response_dict}

    with open(output_data_path, "a", encoding="utf-8") as f:
        json.dump(data_with_conversation, f, ensure_ascii=False)