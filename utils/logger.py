import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """
    로그 설정을 초기화하는 함수.
    
    Parameters:
        - name: 로거 이름
        - log_file: 로그를 저장할 파일 경로
        - level: 로그 레벨 (INFO, DEBUG, ERROR 등)
    
    Returns:
        - logger: 설정된 logger 객체
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(log_file='app.log', level=logging.INFO):
    """
    공통으로 사용되는 로거를 반환하는 함수.
    """
    return setup_logger('TableQA-Evaluator', log_file, level)
