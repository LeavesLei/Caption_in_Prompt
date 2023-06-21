import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:9001/v1"

model = "vicuna-13b-v1.3"
category = 'tuna'
caption = 'A photo of tuna. A tuna on the beach.'
num_rewrite_captions = 5
prompt = f"This is the image caption about {category} category, please refine and rewrite it to {num_rewrite_captions} more diverse and informative caption candidates.\n" +\
         f"#Caption\n{caption}\n" +\
         "#Answer\n"


# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=512)
# print the completion
print(prompt + completion.choices[0].text)