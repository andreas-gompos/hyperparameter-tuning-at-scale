# Dummy Model training locally, on Polyaxon and on SageMaker

## Train Locally

```bash
#!/bin/bash
pip install -r requirements.txt
python ./src/get_data.py --train_channel=/{{ train_channel }}
python ./src/local_train.py --penalty={{ penalty }}
                            --C={{ C }}
                            --train_channel={{ train_channel }}
                            --model_dir={{ model_dir }}
```

| Parameter     | Description                           | Valid Values | Default |
| ------------- | ------------------------------------- | ------------ | ------- |
| C             | Intensity of regularisation           | float        | 1.0     |
| penalty       | Penalty to be used for regularisation | l1, l2       | l2      |
| train_channel | Local directory of training data      | str          | -       |
| model_dir     | Local directory to export the model   | str          | -       |

## Polyaxon

### Login to Polyaxon

```bash
pip install -U polyaxon-cli
polyaxon config set --host=****** --port=******
polyaxon login --username=****** --password=******
```

Validate you are logged in: `polyaxon cluster`

### Train on Polyaxon

> #### Assumptions for the following to work
>
> 1. You need to have a polyaxon cluster running


```bash
- create a project
`polyaxon project create --name=project-1`

- initialise the project
`polyaxon init project-1`

- download the data to the cluster
`polyaxon run -f polyaxonfiles/data.yml -u`

- Upload the code to polyaxon and run experiments
`polyaxon run -f polyaxonfiles/cpu.yml`

- See how much resourses experiment `3` is using:
`polyaxon experiment -xp 3 resources`

- Start a jupyter notebook
`polyaxon notebook start -f polyaxonfiles/notebook.yml`
```

## SageMaker
### High Level Workflow

With SageMaker first you need to create a docker image holding the training environment along with the training code on ECR. The training data lives in S3.
On runtime, SageMaker downloads the docker image from ECR and the training data from S3. Therefore, you need to have your training image in ECR, your data in S3, provide those paths to SageMaker configs and provide a role that has access to all these resources.


> #### Assumptions for the following to work
>
> 1. Generate the dummy training data using the `get_data.py` script
> 2. Upload the training data to a S3 bucket
> 3. In `sagemaker/create_hp_job.py` update:
>
>     - the S3 bucket
>     - the ECR repo
>     - the RoleArn

### Build and Push Docker Image for Training

```bash
docker build -f sagemaker/dockerfiles/train.Dockerfile -t sm_train .
docker tag sm_train aws_account_id.dkr.ecr.region.amazonaws.com/ecr_repo_name:tag
docker push aws_account_id.dkr.ecr.region.amazonaws.com/ecr_repo_name:tag
```

### Train

```bash
python sagemaker/create_hp_job.py --tuning_job_name={{ tuning_job_name }}
```
