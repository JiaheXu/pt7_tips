

from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
model_path = "./"

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto").cuda()

user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"



############################# to markdown #############################
# single-image prompt
# prompt = f"{user_prompt}<|image_1|>\nIs this a real human?{prompt_suffix}{assistant_prompt}"
prompts = []
prompts.append( f"{user_prompt}<|image_1|>\nIs this a real human?{prompt_suffix}{assistant_prompt}" )
prompts.append( f"{user_prompt}<|image_1|>\nCan you see both of his arms?{prompt_suffix}{assistant_prompt}" )
prompts.append( f"{user_prompt}<|image_1|>\nAre his arms injured?{prompt_suffix}{assistant_prompt}" )
prompts.append( f"{user_prompt}<|image_1|>\nCan you see both of his legs?{prompt_suffix}{assistant_prompt}" )
prompts.append( f"{user_prompt}<|image_1|>\nAre his legs injured?{prompt_suffix}{assistant_prompt}" )


file = open('./test/answer.txt', 'r')
answer = file.readlines()
cases = 13

for count1, i in enumerate( range(1,cases+1)):
    image = Image.open(r"./test/test{idx}.png".format(idx = i) )
    correct = 0
    for count2, prompt in enumerate(prompts):
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

        # print(f">>> Prompt\n{prompt}")
        generate_ids = model.generate(**inputs, 
                                    max_new_tokens=1000,
                                    eos_token_id=processor.tokenizer.eos_token_id,
                                    )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
                                        skip_special_tokens=False, 
                                        clean_up_tokenization_spaces=False)[0]
        result = '2'
        if(response[0:3] == "Yes"):
            result = '1'
        if(response[0:2] == "No"):
            result = '0'
    
        if(result == answer[count1][count2]):
            correct += 1
        print(result, end = '')
        
    print(" correct: ", correct)
