import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
    GORQ_API_KEY = os.getenv('GORQ_API_KEY')
    DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')
    RESULTS_PATH = os.getenv('RESULTS_PATH', 'results/')
    DATASET_PATH = os.getenv('DATASET_PATH', 'data/')
    LOG_DIR = os.getenv('LOG_DIR', 'logs/')