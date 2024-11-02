import mlflow

def get_model_data_uri(
        params_to_retrieve: list = None, 
        run_number: int = 0, 
        mlflow_data_path: str = "sqlite:///mlflow/mlflow.db"
    ) -> dict:
    """
    Retrieves model data URI and other specified parameters from an MLflow run.

    Args:
    - params_to_retrieve (list, optional): A list of parameters to retrieve from the MLflow run. Defaults to None.
    - run_number (int, optional): The index of the run to retrieve parameters from. Defaults to 0.
    - mlflow_data_path (str, optional): The path to the MLflow database. Defaults to "sqlite:///mlflow/mlflow.db".

    Returns:
    - dict: A dictionary containing the retrieved parameters.
    """
    mlflow.set_tracking_uri(mlflow_data_path)

    # Get the list of runs, sorted by start time in descending order
    runs = mlflow.search_runs(order_by=["start_time DESC"])

    # Check if there are any runs
    if runs.empty:
        raise ValueError("No runs found in the specified MLflow database.")

    # Select the specified run
    selected_run_id = runs.iloc[run_number].run_id

    # Get the parameters of the selected run
    selected_run_params = mlflow.get_run(selected_run_id).data.params

    # If no specific parameters are provided, default to model_data_uri
    if params_to_retrieve is None:
        params_to_retrieve = ['model_data_uri']

    # Retrieve the requested parameters
    retrieved_params = {param: selected_run_params.get(param) for param in params_to_retrieve}

    return retrieved_params