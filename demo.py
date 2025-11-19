from openai import OpenAI, omit
import json

client = OpenAI(base_url="http://vm-llama.eecs.tufts.edu:8080/v1",
api_key='None')

# response = client.chat.completions.create(
# model="Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
# messages=[
# {"role": "system", "content": "You are a helpful assistant."},
# {"role": "user", "content": "Explain the Tufts Jumbo mascot in one paragraph."}
# ])


import json

def get_shopping_list_shelves(state, player_index=0):
    """
    Return a list of shelf dicts that correspond to items
    in the given player's shopping_list.
    """
    obs = state.get("observation", {})
    players = obs.get("players", [])
    shelves = obs.get("shelves", [])

    if not players:
        return []

    player = players[player_index]
    shopping_list = set(player.get("shopping_list", []))

    # Match either by `food_name` or `food`
    matching_shelves = [
        shelf for shelf in shelves
        if shelf.get("food_name") in shopping_list
        or shelf.get("food") in shopping_list
    ]

    return matching_shelves


def get_registers(state):
    """
    Return registers info.
    """
    obs = state.get("observation", {})

    registers = obs.get("registers", [])

    return {
        "registers": registers
    }


def remove_image_fields(state):
    """Recursively remove keys containing 'image' from dictionaries and lists."""
    if isinstance(state, dict):
        return {
            key: remove_image_fields(value)
            for key, value in state.items()
            if "image" not in key.lower()  # remove keys like food_image, shelf_image, imagePath, etc.
        }
    elif isinstance(state, list):
        return [remove_image_fields(item) for item in state]
    else:
        return state


def clean_json_data(input_path):
    # Open JSOn file and read input
    with open(input_path, "r") as f:
        data = json.load(f)

    # Clean the JSON object
    cleaned = remove_image_fields(data)
    registers = get_registers(cleaned)
    cleaned = get_shopping_list_shelves(cleaned)

    # Write cleaned JSON to output file
    with open("observation_cleaned.json", "w") as f:
        json.dump(cleaned, f, indent=4)


import json

def get_shopping_list_shelves(state, player_index=0):
    """
    Return a list of shelf dicts that correspond to items
    in the given player's shopping_list.
    """
    obs = state.get("observation", {})
    players = obs.get("players", [])
    shelves = obs.get("shelves", [])

    if not players:
        return []

    player = players[player_index]
    shopping_list = set(player.get("shopping_list", []))

    # Match either by `food_name` or `food`
    matching_shelves = [
        shelf for shelf in shelves
        if shelf.get("food_name") in shopping_list
        or shelf.get("food") in shopping_list
    ]

    return matching_shelves


def get_registers(state):
    """
    Return registers info.
    """
    obs = state.get("observation", {})

    registers = obs.get("registers", [])

    return {
        "registers": registers
    }


def remove_image_fields(state):
    """Recursively remove keys containing 'image' from dictionaries and lists."""
    if isinstance(state, dict):
        return {
            key: remove_image_fields(value)
            for key, value in state.items()
            if "image" not in key.lower()  # remove keys like food_image, shelf_image, imagePath, etc.
        }
    elif isinstance(state, list):
        return [remove_image_fields(item) for item in state]
    else:
        return state


def clean_json_data(input_path):
    # Open JSOn file and read input
    with open(input_path, "r") as f:
        data = json.load(f)

    # Clean the JSON object
    cleaned = remove_image_fields(data)
    registers = get_registers(cleaned)
    cleaned = get_shopping_list_shelves(cleaned)

    # Write cleaned JSON to output file
    with open("observation_cleaned.json", "w") as f:
        json.dump(cleaned, f, indent=4)


def run_llm_query(prompt_index = 3, model = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"):
    with open(f'prompts/prompt{prompt_index}.txt', "r", encoding="utf-8") as file:
        prompt = file.read()
        print(prompt)
    print('running llm -----------------------')
    # sys = 'You are a supervising agent. You will mediate agents that are completing shopping tasks in a mall when they violate norms.'
    sys = 'You are MediatorAgent, a supervising LLM responsible for resolving norm violations between shopping agents in a mall environment. Your only task is to produce a valid action plan for one agent that resolves the conflict.'
    # print(sys)
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt},
    ],
    temperature=0)
    print('getting response:')
    response = response.choices[0].message.content.strip("\n")
    return response.split()








if __name__ == '__main__':
    # with open('prompts/prompt2.txt', "r", encoding="utf-8") as file:
    #     prompt = file.read()
    #     print(prompt)
    # # Example usage of run_gpt4_query
    # response = run_gpt4_query(prompt)
    # # print(response)

    # clean_json_data("observation.json")
    # print(run_gpt4_query("Is Cinderella appropriate for kids of age 5yo to read?"))
    print(run_llm_query())