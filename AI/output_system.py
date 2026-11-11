import os
import time
import logging

# Configure logger
logger = logging.getLogger('FileOperationsMonitor')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('ai.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Workspace directory
WORKSPACE_DIR = 'workspace'

# Ensure the workspace directory exists
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

def log_operation(operation, file_path, duration):
    logger.debug(f'Operation: {operation}, File: {file_path}, Duration: {duration:.6f} seconds')

def create_file(file_path, content=''):
    start_time = time.perf_counter()
    with open(file_path, 'w') as file:
        file.write(content)
    end_time = time.perf_counter()
    log_operation('Create', file_path, end_time - start_time)

def read_file(file_path):
    start_time = time.perf_counter()
    with open(file_path, 'r') as file:
        content = file.read()
    end_time = time.perf_counter()
    log_operation('Read', file_path, end_time - start_time)
    return content

def update_file(file_path, content):
    start_time = time.perf_counter()
    with open(file_path, 'a') as file:
        file.write(content)
    end_time = time.perf_counter()
    log_operation('Update', file_path, end_time - start_time)

def delete_file(file_path):
    start_time = time.perf_counter()
    os.remove(file_path)
    end_time = time.perf_counter()
    log_operation('Delete', file_path, end_time - start_time)

if __name__ == "__main__":
    # Sample file path
    sample_file_path = os.path.join(WORKSPACE_DIR, 'sample_data.txt')
    
    # Create a mock data file
    print("Step 1: Creating a mock data file...")
    create_file(sample_file_path, "This is a sample data.\n")
    
    # Read the mock data file
    print("Step 2: Reading the mock data file...")
    content = read_file(sample_file_path)
    print("Content read:", content)
    
    # Update the mock data file
    print("Step 3: Updating the mock data file...")
    update_file(sample_file_path, "Adding another line to the sample data.\n")
    
    # Read the updated mock data file
    print("Step 4: Reading the updated mock data file...")
    updated_content = read_file(sample_file_path)
    print("Updated content read:", updated_content)
    
    # Delete the mock data file
    print("Step 5: Deleting the mock data file...")
    delete_file(sample_file_path)
    
    print("All operations completed and logged.")