import time
import mlflow
import sagemaker
from sagemaker.huggingface import HuggingFace

from .config import AWSConfig, S3Config, ModelConfig
from .utils import AWSConnector
from ...utils.logger import setup_logger

logger = setup_logger("model_trainer", "model_training.log")

class SageMakerTraining:
    def __init__(self, aws_connector, sagemaker_role):
        logger.info("Initializing SageMakerTraining...")
        self.sagemaker_session = aws_connector.sagemaker_session
        self.boto3_session = aws_connector.boto3_session
        self.sagemaker_role = sagemaker_role
    
    def create_job_name(self) -> str:
        return f'falcon-qlora-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
    
    def get_hyperparameters(
            self, 
            training_input_path: str, 
            testing_input_path: str, 
            epochs: int = 1, 
            batch_size: int = 2, 
            learning_rate: float = 2e-4
        ) -> dict:
        return {
            'model_id': ModelConfig.MODEL_NAME,
            'train_data_path': training_input_path,
            'val_data_path': testing_input_path,
            'epochs': epochs,
            'per_device_train_batch_size': batch_size,
            'lr': learning_rate,
        }
    
    def create_estimator(self, job_name, hyperparameters):
        logger.info("Creating HuggingFace estimator...")
        return HuggingFace(
            entry_point='train.py',
            source_dir='training_scripts',
            instance_type='ml.p3.2xlarge',
            instance_count=1,
            base_job_name=job_name,
            role=self.sagemaker_role,
            volume_size=300,
            transformers_version='4.28',
            pytorch_version='2.0',
            py_version='py310',
            hyperparameters=hyperparameters,
            use_spot_instances=True,
            max_run=1 * 60 * 60,
            max_wait=2 * 60 * 60,
            environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"},
            dependencies=['src/training_scripts/requirements.txt'],
        )
    
    def run_training(self, training_input_path, testing_input_path):
        logger.info("Starting training process...")
        job_name = self.create_job_name()
        hyperparameters = self.get_hyperparameters(
            training_input_path, 
            testing_input_path
        )

        mlflow.set_tracking_uri("C:\Users\RaviB\GitHub\vegan-ai-nutritionist\mlflow")
        
        with mlflow.start_run():
            logger.info("Starting MLflow run...")
            estimator = self.create_estimator(job_name, hyperparameters)
            
            data = {
                'train': training_input_path,
                'test': testing_input_path
            }
            
            logger.info("Fitting the estimator...")
            estimator.fit(data, wait=True)

            model_data_uri = estimator.model_data
            logger.info(f"Model training completed. Model data URI: {model_data_uri}")

            #save the model_data_uri for deployment later
            mlflow.log_param("model_data_uri", model_data_uri)

            #save the parameters here too
            mlflow.log_param("model_id", ModelConfig.MODEL_NAME)
            mlflow.log_param("job_name", job_name)
            mlflow.log_param("epochs", hyperparameters['epochs'])
            mlflow.log_param("batch_size", hyperparameters['per_device_train_batch_size'])
            mlflow.log_param("learning_rate", hyperparameters['lr'])
            logger.info("Training parameters logged.")

            return estimator, model_data_uri
    
def main():
    logger.info("Loading AWS connections...")
    aws_connector = AWSConnector()
    
    training_input_path = S3Config.get_train_data_uri()
    testing_input_path = S3Config.get_test_data_uri()

    sage_maker_training = SageMakerTraining(aws_connector, AWSConfig.SAGEMAKER_ROLE)

    logger.info("Running training...")
    _ = sage_maker_training.run_training(training_input_path, testing_input_path)

if __name__ == "__main__":
    main()