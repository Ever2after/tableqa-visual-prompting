from openai import OpenAI
from config import Config
import json

openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

class GPT:
    def __init__(self, model='gpt-4o-mini', response_format="text", temperature=0.1):
        self.model = model
        self.response_format = response_format
        self.temperature = temperature

    def get_gpt(self, messages):
        try:
            response = openai_client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                response_format={'type': self.response_format},
            )
            answer = response.choices[0].message.content.strip()
            return answer if self.response_format == 'text' else json.loads(answer)
        except Exception as e:
            print(f"GPT Error: {e}")
            return ""

    def get_tableqa_answer(self, mode='text', table=None, query=None):
        if mode == 'text':
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer the following question. Just give the answer, not the process. \nTable: "},
                    {"type": "text", "text": table},
                    {"type": "text", "text": f"Question: {query}\nAnswer:"}
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
                    {"type": "text", "text": f"Question: {query}\nAnswer:"}
                ],
            }]
        else:
            return ""
        
        answer = self.get_gpt(messages)
        if answer:
            return answer.lower()
        return ""
