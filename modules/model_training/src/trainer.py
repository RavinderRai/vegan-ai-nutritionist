import time
import sagemaker
from sagemaker.huggingface import HuggingFace

class SageMakerTraining:
    def __init__(self, session, role):
        self.session = session
        self.role = role
    
    def create_job_name(self):
        return f'falcon-qlora-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
    
    def get_hyperparameters(self, model_id, training_input_path, testing_input_path):
        return {
            'model_id': model_id,
            'train_data_path': training_input_path,
            'val_data_path': testing_input_path,
            'epochs': 1,
            'per_device_train_batch_size': 2,
            'lr': 2e-4,
        }
    
    def create_estimator(self, job_name, hyperparameters):
        return HuggingFace(
            entry_point='train.py',
            source_dir='notebooks/scripts',
            instance_type='ml.p3.2xlarge',
            instance_count=1,
            base_job_name=job_name,
            role=self.role,
            volume_size=300,
            transformers_version='4.28',
            pytorch_version='2.0',
            py_version='py310',
            hyperparameters=hyperparameters,
            use_spot_instances=True,
            max_run=1 * 60 * 60,
            max_wait=2 * 60 * 60,
            environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"},
            dependencies=['notebooks/scripts/requirements.txt'],
        )
    
    def run_training(self, model_id, training_input_path, testing_input_path):
        job_name = self.create_job_name()
        hyperparameters = self.get_hyperparameters(
            model_id, 
            training_input_path, 
            testing_input_path
        )
        
        estimator = self.create_estimator(job_name, hyperparameters)
        
        data = {
            'train': training_input_path,
            'test': testing_input_path
        }
        
        estimator.fit(data, wait=True)
        return estimator