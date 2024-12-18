import re
import logging
from sagemaker.huggingface import HuggingFacePredictor

# change back to
# from .model_utils import get_model_data_uri
# from ...config import AWSConfigCredentials 
# from ...utils.utils import AWSConnector
# if not running strealit app
from inference.src.model_utils import get_model_data_uri
from config import AWSConfigCredentials
from utils.utils import AWSConnector

class ModelInference:
    """
    A class for performing inference using a Hugging Face model deployed on Amazon SageMaker.
    """
    
    def __init__(self, endpoint_name: str = None):
        """
        Initializes the ModelInference class with the specified SageMaker endpoint.
        
        Args:
            endpoint_name (str): The name of the SageMaker endpoint to be used for inference.
        """
        logging.basicConfig(level=logging.INFO)
        
        if endpoint_name is None:
            latest_endpoint = get_model_data_uri(
                ['endpoint_name'], 
                run_number=0, 
                mlflow_data_path="sqlite:///C:/Users/RaviB/GitHub/vegan-ai-nutritionist/mlflow/endpoints.db"
            )
            endpoint_name = latest_endpoint['endpoint_name']            
        
        self.aws_connector = AWSConnector()
        self.aws_credentials = AWSConfigCredentials.load()
        
        self.sagemaker_session=self.aws_connector.sagemaker_session
        
        self.predictor = HuggingFacePredictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self.sagemaker_session
        )
    
    def predict(self, prompt: str):
        """
        Generates a response for the given prompt using the deployed model.
        
        Args:
            prompt (str): The input prompt to generate a response for.
        
        Returns:
            str: The generated response from the model or an error message if the prediction fails.
        """
        request = {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.7,
                "max_new_tokens": 512,
                "stop": ["\nUser:","<|endoftext|>","</s>"]
            }
        }
        
        try:
            response = self.predictor.predict(request)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return "An error occurred while generating the response."
        
        assistant_response = response[0]['generated_text']
        
        return self.extract_falcon_response(assistant_response)
    
    def extract_falcon_response(self, output: str) -> str:
        """
        Extracts the response generated by Falcon from the model output.
        
        Args:
            output (str): The complete output generated by the model.
        
        Returns:
            str: The extracted Falcon response or the full output if extraction fails.
        """
        match = re.search(r'\nFalcon:(.*?)\nUser', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        logging.warning("Falcon response not found in output.")
        return output