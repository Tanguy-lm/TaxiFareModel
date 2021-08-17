import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "[FR][Nice][Tanguy-lm]TaxiFareModel v1"

# Indicate mlflow to log to remote server
mlflow.set_tracking_uri("https://mlflow.lewagon.co/")

client = MlflowClient()

try:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
except BaseException:
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

yourname = 'Tanguy'

if yourname is None:
    print("please define your name, il will be used as a parameter to log")

for model in ["linear", "Randomforest"]:
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "rmse", 4.5)
    client.log_param(run.info.run_id, "model", model)
    client.log_param(run.info.run_id, "student_name", yourname)
