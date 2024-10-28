from datasets import Dataset
from transformers import AutoTokenizer
from .config import ModelConfig, S3Config
from .utils import AWSConnector

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
    
    def process_and_save_datasets(self, train_data, test_data, sagemaker_session):
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
        bucket = sagemaker_session.default_bucket()
        training_path = f's3://{bucket}/processed/falcon/tokenized-train-data'
        testing_path = f's3://{bucket}/processed/falcon/tokenized-test-data'
        
        tokenized_train.save_to_disk(training_path)
        tokenized_test.save_to_disk(testing_path)
        
        return training_path, testing_path
    
def main():
    # Initialize AWS connection
    aws_connector = AWSConnector()
    session_info = aws_connector.get_session_info()
    print(f"SageMaker Role ARN: {session_info['role_arn']}")
    print(f"SageMaker Bucket: {session_info['bucket']}")
    print(f"SageMaker Session Region: {session_info['region']}")
    
    # Load data from S3
    train_data = aws_connector.load_data_from_s3(
        S3Config.BUCKET_NAME, 
        S3Config.TRAIN_DATA_KEY
    )
    test_data = aws_connector.load_data_from_s3(
        S3Config.BUCKET_NAME, 
        S3Config.TEST_DATA_KEY
    )
    
    # Process and save datasets
    processor = Tokenizer()
    train_path, test_path = processor.process_and_save_datasets(
        train_data, 
        test_data, 
        aws_connector.sagemaker_session
    )
    
    print("Uploaded data to:")
    print(f"Training dataset: {train_path}")
    print(f"Testing dataset: {test_path}")

if __name__ == "__main__":
    main()