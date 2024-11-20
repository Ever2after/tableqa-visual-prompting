from config import Config
import requests
import json

class Gemini:
    def __init__(self, model='gemini-1.5-flash-latest'):
        self.model = model

    def get_response(self, messages):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={Config.GOOGLE_AI_API_KEY}"
        request_data = { 
            "contents": [{
                    "parts": messages
                }]}
        headers = { 'Content-Type': 'application/json' }
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(request_data)
            )
            response = response.json()
            return response['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            print(f"Gemini Error: {e}")
            return ""

    def get_tableqa_answer(self, mode='text', table=None, query=None, cot=False):
        if mode == 'text':
            messages = [
                {"text": "Answer the following question. Just give the answer, not the process. \nTable: "},
                {"text": table},
                {"text": f"Question: {query}\nAnswer:"} \
                    if not cot else {"text": f"Question: {query}\nLet's think step by step, and then give the final answer.\nAnswer: "},
            ]
        elif mode == 'image':
            messages = [
                {"text": "Answer the following question. Just give the answer, not the process. \nTable: "},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": table
                    }
                },
                {"text": f"Question: {query}\nAnswer:"} \
                    if not cot else {"text": f"Question: {query}\nLet's think step by step, and then give the final answer.\nAnswer: "},
            ]
        else:
            return ""
        
        answer = self.get_response(messages)
        if answer:
            return answer.lower()
        return ""