from openai import OpenAI
from config import Config
import json

openai_client = OpenAI(
    api_key=Config.DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai"
)

class Llama:
    def __init__(self, model='meta-llama/Llama-3.2-11B-Vision-Instruct', response_format='text', temperature=0.1, max_tokens=2000):
        self.model = model
        self.response_format = response_format
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_response(self, messages):
        try:
            response = openai_client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            answer = response.choices[0].message.content.strip()
            return answer if self.response_format == 'text' else json.loads(answer)
        except Exception as e:
            print(f"Llama Error: {e}")
            return ""

    def get_tableqa_answer(self, mode='text', table=None, query=None, cot=False):
        if mode == 'text':
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer the following question. Just give the answer, not the process. \nTable: "},
                    {"type": "text", "text": table},
                    {"type": "text", "text": f"Question: {query}\nAnswer:"} \
                        if not cot else {"type": "text", "text": f"Question: {query}\nLet's think step by step, and then give the final answer.\nAnswer: "},
                ],
            }]
        elif mode == 'image':
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer the following question. Just give the answer, not the process. \nTable: "},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url" : f"data:image/jpeg;base64, {table}",
                            "detail" : 'auto'
                        }
                    },
                    {"type": "text", "text": f"Question: {query}\nJust give only the answer, not the process.\nAnswer:"} \
                        if not cot else {"type": "text", "text": f"Question: {query}\nLet's think step by step, and then give the final answer.\nAnswer: "},
                ],
            }]
        else:
            return ""
        
        answer = self.get_response(messages)
        if answer:
            return answer.lower()
        return ""
