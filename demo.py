from openai import OpenAI, omit


client = OpenAI(base_url="http://vm-llama.eecs.tufts.edu:8080/v1",
api_key='None')

# response = client.chat.completions.create(
# model="Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
# messages=[
# {"role": "system", "content": "You are a helpful assistant."},
# {"role": "user", "content": "Explain the Tufts Jumbo mascot in one paragraph."}
# ])


def run_gpt4_query(prompt, model = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"):
    print('running gpt -----------------------')
    sys = 'You are a supervising agent. You will mediate agents that complete shopping tasks in a mall.'
    # print(sys)
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt},
    ],
    temperature=0)
    print('getting response:')
    response = response.choices[0].message.content.strip("\n")
    return response


if __name__ == '__main__':
    response = run_gpt4_query("How are you doing?")
    print(response)