import logging
import os
from datetime import datetime
from src.config import LOGS_DIR


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(LOGS_DIR, LOG_FILE)


os.makedirs(LOGS_DIR, exist_ok=True)


logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)



