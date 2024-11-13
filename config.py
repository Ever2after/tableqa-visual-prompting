import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
    RESULTS_PATH = os.getenv('RESULTS_PATH', 'results/')
    DATASET_PATH = os.getenv('DATASET_PATH', 'data/')