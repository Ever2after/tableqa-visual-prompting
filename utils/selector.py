from model import gpt, gemini, llama
from config import Config
from utils.logger import setup_logger

logger = setup_logger("Selector", log_file='selector.log')

class LLMSelector:
    """
    LLM 기반 선택기 클래스.
    """
    
    def __init__(self, model_name='gpt', cache_dir=None):
        self.cache_dir = cache_dir or Config.DATASET_PATH
        self.mode_name = model_name
        self.model = self._get_model_class(model_name)
    
    def _get_model_class(self, model_name):
        """
        모델 이름에 따라 적절한 모델 클래스 반환.
        """
        model_classes = {
            'gpt': gpt.GPT,
            'gemini': gemini.Gemini,
            'llama': llama.Llama
        }
        if model_name not in model_classes:
            raise ValueError(f"Unsupported model: {model_name}")
        return model_classes[model_name]
    
    def select(self, table, query, candidates):
        candidates_input = "\n".join([f"{i}. {c}" for i, c in enumerate(candidates)])
        prompt = f"""
You will get a table, a question, and {len(candidates)} answer candidates.
Based on this table, choose the more correct answer from the candidates.
If candidatss are the same, choose first one.
If neither response is correct, choose first one.
If the table does not have enough information, choose first one.
Let's think step by step, and then give the final answer.
Final answer should be formatted as 'Final Answer Index: i'
[TABLE] 
{table}
[QUESTION] {query}
[CANDIDATES]
{candidates_input}
"""
        model = self.model()
        if self.mode_name == 'gpt':
            messages = [{"role": "user", "content": prompt}]
        elif self.mode_name == 'gemini':
            messages = [{"text": prompt }]
        elif self.mode_name == 'llama':
            messages = [{"role": "user", "content": prompt}]
        answer = model.get_response(messages).lower()
        # parse index from answer
        try:
            index = int(answer.split("answer index:")[-1].strip().replace("*", ""))
            if index >= len(candidates):
                return 0
            return index
        except:
            logger.error(f"Failed to parse answer: {answer}")
            return 0
    
    

    


