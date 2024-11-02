from datasets import Dataset
from transformers import AutoTokenizer
import logging
from ...config import ModelConfig, S3Config
from ...utils.utils import AWSConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def convert_to_dict_format(self, data):
        return {key: [d[key] for d in data] for key in data[0]}
    
    def tokenize_function(self, examples):
        # Combine 'about_me' and 'context' for the full context
        full_context = examples['about_me'] + ' ' + examples['context']
        
        # Create the prompt using the full context
        prompt = f"Question: {examples['question']}\nContext: {full_context}\nAnswer:"
        response = examples['response']
        
        # Tokenize inputs and labels
        tokenized_input = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
        tokenized_output = self.tokenizer(response, truncation=True, padding="max_length", max_length=512)
        
        # Combine input and output (GPT-2 is autoregressive)
        input_ids = tokenized_input["input_ids"] + tokenized_output["input_ids"]
        
        # Create the labels (output sequence should be the entire concatenated sequence)
        labels = [-100] * len(tokenized_input["input_ids"]) + tokenized_output["input_ids"]
        
        return {
            "input_ids": input_ids, 
            "attention_mask": [1] * len(input_ids), 
            "labels": labels
        }
    
    def process_and_save_datasets(self, train_data, test_data, tokenized_training_path, tokenized_test_path):
        # Convert to dictionary format
        train_dict = self.convert_to_dict_format(train_data)
        test_dict = self.convert_to_dict_format(test_data)
        
        # Create datasets
        train_dataset = Dataset.from_dict(train_dict)
        test_dataset = Dataset.from_dict(test_dict)
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            self.tokenize_function, 
            remove_columns=train_dataset.column_names
        )
        tokenized_test = test_dataset.map(
            self.tokenize_function, 
            remove_columns=test_dataset.column_names
        )
        
        # Save to S3
        #training_path = f's3://{bucket_name}/tokenized-data/falcon/tokenized-train-data'
        #testing_path = f's3://{bucket_name}/tokenized-data/falcon/tokenized-test-data'
        
        tokenized_train.save_to_disk(tokenized_training_path)
        tokenized_test.save_to_disk(tokenized_test_path)
        
        return tokenized_training_path, tokenized_test_path
    
def main():
    logger.info("Initializing AWS connection..")
    aws_connector = AWSConnector()
    #session_info = aws_connector.get_session_info()
    
    logger.info("Loading data from S3..")
    train_data = aws_connector.load_data_from_s3(
        S3Config.BUCKET_NAME, 
        S3Config.TRAIN_DATA_KEY
    )
    test_data = aws_connector.load_data_from_s3(
        S3Config.BUCKET_NAME, 
        S3Config.TEST_DATA_KEY
    )
    
    logger.info("Saving tokenized datasets..")
    processor = Tokenizer()
    s3_config = S3Config()

    tokenized_training_path, tokenized_test_path = processor.process_and_save_datasets(
        train_data, 
        test_data,
        s3_config.get_tokenized_train_data_uri(),
        s3_config.get_tokenized_test_data_uri()
    )
    
    logger.info("Uploaded data to:")
    logger.info(f"Training dataset: {tokenized_training_path}")
    logger.info(f"Testing dataset: {tokenized_test_path}")

if __name__ == "__main__":
    main()