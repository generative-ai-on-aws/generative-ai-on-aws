import os
from dotenv import load_dotenv

import os
from dotenv import load_dotenv, find_dotenv
import warnings

import requests
import json

# 전역 변수 초기화하기
_ = load_dotenv(find_dotenv())
# warnings.filterwarnings('ignore')
url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/inference"
headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }

def llama_guard(query, 
               model="Meta-Llama/Llama-Guard-7b", 
               temperature=0.0, 
               max_token=1024,
               verbose=False,
               base=2,
               max_tries=3):
    
    prompt = f"[INST]{query}[/INST]"
    
    data = {
      "model": model,
      "prompt": prompt,
      "temperature": temperature,
      "max_tokens": max_token
    }
    if verbose:
        print(f"model: {model}")
        print("Input is wrapped in [INST] [/INST] tags")

    # API 호출 시에 장애가 발생한다면 여러 번 호출 허용합니다.
    # 3회 호출 시도가 모두 실패한 후에 제공된 응답을 사용자에게 반환합니다.
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()['output']['choices'][0]['text']
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])

    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response


def safe_llama(query, add_inst=True, 
               model="togethercomputer/llama-2-7b-chat",
               safety_model="Meta-Llama/Llama-Guard-7b",
               temperature=0.0, max_token=1024,
               verbose=False,
               base=2,
               max_tries=3):
    if add_inst:
        prompt = f"[INST]{query}[/INST]"
    else:
        prompt = query
    
    if verbose:
        print(f"model: {model}")
        print(f"safety_model:{safety_model}")
    
    data = {
      "model": model,
      "prompt": prompt,
      "temperature": temperature,
      "max_tokens": max_token,
      "safety_model": safety_model
    }

    # API 호출 시에 장애가 발생한다면 여러 번 호출 허용합니다.
    # 3회 호출 시도가 모두 실패한 후에 제공된 응답을 사용자에게 반환합니다.
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.json()['output']['choices'][0]['text']
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response           

  

def code_llama(prompt, 
          model="togethercomputer/CodeLlama-7b-Instruct", 
          temperature=0.0, 
          max_tokens=1024,
          verbose=False,
          url=url,
          headers=headers,
          base=2,
          max_tries=3):

    if model.endswith("Instruct"):
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    # API 호출 시에 장애가 발생한다면 여러 번 호출 허용합니다.
    # 3회 호출 시도가 모두 실패한 후에 제공된 응답을 사용자에게 반환합니다.
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()['output']['choices'][0]['text']
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response


# 20은 최소 새 토큰 수, 
# 이는 입력 프롬프트의 최대 토큰 수를 허용합니다: 4097 - 20 = 4077
# 그러나 max_tokens는 추력 토큰 수를 제한합니다.
# 입력 프롬프트 토큰 수 + max_tokens (응답 토큰 수)의 합은 4097을 초과할 수 없습니다.
def llama(prompt, 
          add_inst=True, 
          model="togethercomputer/llama-2-7b-chat", 
          temperature=0.0, 
          max_tokens=1024,
          verbose=False,
          url=url,
          headers=headers,
          base=2, # 대기할 시간(초)
          max_tries=3):
    
    if add_inst:
        prompt = f"[INST]{prompt}[/INST]"

    if verbose:
        print(f"Prompt:\n{prompt}\n")
        print(f"model: {model}")

    data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    # API 호출 시에 장애가 발생한다면 여러 번 호출 허용합니다.
    # 3회 호출 시도가 모두 실패한 후에 제공된 응답을 사용자에게 반환합니다.    
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = requests.post(url, headers=headers, json=data)
            return response.json()['output']['choices'][0]['text']
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response


def llama_chat(prompts, 
               responses,
               model="togethercomputer/llama-2-7b-chat", 
               temperature=0.0, 
               max_tokens=1024,
               verbose=False,
               url=url,
               headers=headers,
               base=2,
               max_tries=3
              ):

    prompt = get_prompt_chat(prompts,responses)

    # API 호출 시에 장애가 발생한다면 여러 번 호출 허용합니다.
    # 3회 호출 시도가 모두 실패한 후에 제공된 응답을 사용자에게 반환합니다.
    wait_seconds = [base**i for i in range(max_tries)]

    for num_tries in range(max_tries):
        try:
            response = llama(prompt=prompt,
                             add_inst=False,
                             model=model, 
                             temperature=temperature, 
                             max_tokens=max_tokens,
                             verbose=verbose,
                             url=url,
                             headers=headers
                            )
            return response
        except Exception as e:
            if response.status_code != 500:
                return response.json()

            print(f"error message: {e}")
            print(f"response object: {response}")
            print(f"num_tries {num_tries}")
            print(f"Waiting {wait_seconds[num_tries]} seconds before automatically trying again.")
            time.sleep(wait_seconds[num_tries])
 
    print(f"Tried {max_tries} times to make API call to get a valid response object")
    print("Returning provided response")
    return response


def get_prompt_chat(queries, responses):
    prompt_chat = f"<s>[INST] {queries[0]} [/INST]"
    for n, response in enumerate(responses):
        query = queries[n + 1]
        prompt_chat += f"\n{response}\n </s><s>[INST] \n{ query }\n [/INST]"
    
    return prompt_chat


policy = """
<BEGIN UNSAFE CONTENT CATEGORIES>
O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
<END UNSAFE CONTENT CATEGORIES>
"""

