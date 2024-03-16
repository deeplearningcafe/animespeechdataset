import logging
import os

log_filename = "src\characterdataset\common\logs\dataset_manager.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

log = logging.getLogger(__name__)
