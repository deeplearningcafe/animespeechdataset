import logging
import os

# Get the absolute path of the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the logs directory
log_filename = os.path.join(current_dir, "logs")
os.makedirs(log_filename, exist_ok=True)

# Define the absolute path to the log file
log_file_path = os.path.join(log_filename, "dataset_manager.log")

logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

log = logging.getLogger(__name__)
logging.getLogger('gradio').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
