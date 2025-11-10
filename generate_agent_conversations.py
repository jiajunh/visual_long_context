import os
import json

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


from openai import OpenAI

OPENAI_API_KEY=""

# def generate_paragraphs(prompt, image_url, model="gpt-5-nano"):
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     response = client.responses.create(
#         model=model,
#         input=[
#             {
#                 "role": "user",
#                 "content": [
#                     { "type": "input_text", "text": prompt},
#                     {
#                         "type": "input_image",
#                         "image_url": image_url,
#                     },
#                 ],
#             }
#         ],
#     )
#     return response.output_text



def generate_paragraphs(prompt, image_url, model="gpt-5-nano"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )
    return response.output_text



def process_item(k, v, prompt, model):
    raw = generate_paragraphs(prompt=prompt, image_url=v["url"], model=model)
    parsed = json.loads(raw)
    return k, {**v, **parsed}


if __name__ == "__main__":
    model = "gpt-5-nano"
    num_paragraph = 10
    paragraph_length = 100

    original_data_path = "./data/samples_500/captions.json"
    output_data_path = f"./data/samples_500/gpt-5-nano_{num_paragraph}_{paragraph_length}.json"

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

    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))  # default smaller because processes are heavier

    with open(original_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_with_conversation = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {
            exe.submit(process_item, key, val, prompt, model): key
            # for key, val in data.items()
            for key, val in list(data.items())[:4]
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating Conversation"):
            key = futures[fut]
            try:
                k, result = fut.result()
                data_with_conversation[k] = result
            except Exception as e:
                data_with_conversation[key] = {"generation_error": str(e)}

    # write out full JSON (overwrite to produce a clean single file)
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    with open(output_data_path, "w", encoding="utf-8") as f:
        json.dump(data_with_conversation, f, ensure_ascii=False, indent=2)

    print(f"Finished. Wrote {len(data_with_conversation)} items to {output_data_path}")



# if __name__ == "__main__":
#     original_data_path = "./data/samples_500/captions.json"
#     output_data_path = "./data/samples_500/agent_conversation.json"

#     model = "gpt-5-nano"
#     num_paragraph = 10
#     paragraph_length = 100

#     prompt = f"""Construct three long context start with the image, as descriptions for every paragraph.\
#     Each context should contain about {num_paragraph} paragraphs, and each paragraph is about {paragraph_length} words.\
#     Paragraph 1 in each context must start with something highly relevant to the image.\
#     As the paragraphs progress, the topic may gradually expand to related ideas,\
#     but each new paragraph should still maintain at least a loose connection to something previously mentioned,\
#     even if the connection becomes weaker.\
#     The first context should stay almost most focused on topic related to the image.\
#     Almost every paragraph should relate to the image or its direct concepts.\
#     The second context should be more diverse and may go beyond the image.\
#     Later paragraphs may explore broader ideas, but must still connect to elements introduced earlier.\
#     The third context should be the most creative and diverse. Paragraphs can shift topics boldly,\
#     but every transition must still trace back to something mentioned before.\
#     Return the response in a json format. The keys are 'difficulty_1', 'difficulty_2' and 'difficulty_3'\
#     Each value is a list of the paragraphs."""

#     print(prompt)

#     with open(original_data_path) as f:
#         data = json.load(f)

#     data_with_conversation = {}

#     for key, val in tqdm(data.items(), desc="Generating Conversation"):
#         response = generate_paragraphs(prompt=prompt, image_url=val["url"], model="gpt-5")
#         response_dict = json.loads(response)
#         data_with_conversation[key] = {**val, **response_dict}


#     os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
#     with open(output_data_path, "a", encoding="utf-8") as f:
#         json.dump(data_with_conversation, f, ensure_ascii=False)




