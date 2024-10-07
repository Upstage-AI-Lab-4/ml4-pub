# monitoring.py

import logging

# 로깅 설정
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)