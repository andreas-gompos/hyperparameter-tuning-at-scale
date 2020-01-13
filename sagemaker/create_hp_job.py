import argparse
import subprocess

import json


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_job_name", type=str, required=True)
    return parser.parse_known_args()[0]


def get_training_job_definition():

    training_job_definition = {
        "AlgorithmSpecification": {
            "MetricDefinitions": [
                {"Name": "cv_roc_auc", "Regex": "cv_roc_auc: (.*?);"},
                {"Name": "test_roc_auc", "Regex": "test_roc_auc: (.*?);"},
                {"Name": "test_logloss", "Regex": "test_logloss: (.*?);"},
            ],
            "TrainingImage": "086987898798.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-test:v7",
            "TrainingInputMode": "File",
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://datagusto-sagemaker-eu-west-1/data/train/",
                    }
                },
            },
            {
                "ChannelName": "test",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://datagusto-sagemaker-eu-west-1/data/test/",
                    }
                },
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": "s3://datagusto-sagemaker-eu-west-1/outputs/"
        },
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.large",
            "VolumeSizeInGB": 5,
        },
        "RoleArn": "arn:aws:iam::086987898798:role/service-role/AmazonSageMaker-ExecutionRole-20200103T170367",
        "StaticHyperParameters": {"string": "string"},
        "StoppingCondition": {"MaxRuntimeInSeconds": 7200},
    }
    return training_job_definition


def get_tuning_job_config():

    tuning_job_config = {
        "HyperParameterTuningJobObjective": {
            "MetricName": "cv_roc_auc",
            "Type": "Maximize",
        },
        "ParameterRanges": {
            "CategoricalParameterRanges": [{"Name": "penalty", "Values": ["l1", "l2"]}],
            "ContinuousParameterRanges": [
                {
                    "Name": "C",
                    "MaxValue": "10",
                    "MinValue": "0.01",
                    "ScalingType": "Logarithmic",
                }
            ],
        },
        "ResourceLimits": {"MaxNumberOfTrainingJobs": 3, "MaxParallelTrainingJobs": 1},
        "Strategy": "Random",
        "TrainingJobEarlyStoppingType": "Off",
    }
    return tuning_job_config


def create_hp_job(tuning_job_name):
    training_job_definition = get_training_job_definition()
    tuning_job_config = get_tuning_job_config()

    command = f"""
    aws sagemaker create-hyper-parameter-tuning-job \
        --hyper-parameter-tuning-job-name '{tuning_job_name}' \
        --hyper-parameter-tuning-job-config '{json.dumps(tuning_job_config)}' \
        --training-job-definition '{json.dumps(training_job_definition)}'
    """

    subprocess.check_call(command, shell=True)


def main():
    create_hp_job(ARGS.tuning_job_name)


if __name__ == "__main__":

    ARGS = parse_arguments()
    main()
