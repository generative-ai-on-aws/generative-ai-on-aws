# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Helper utilities for working with Amazon Bedrock from Python notebooks"""
# Python Built-Ins:
import os
from typing import Optional
import json
from datetime import datetime

# External Dependencies:
import boto3
from botocore.config import Config

def get_bedrock(
    assumed_role: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides
    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock = session.client(
        service_name="bedrock",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock._endpoint)
    return bedrock


## This function will help you invoke the model and measure latency
def call_bedrock(bedrock_client, prompt_data, modelId = "anthropic.claude-v2", max_tokens = 4096, temperature = 0, top_p = 0.9):
    if 'amazon' in modelId:
        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig":
            {
                "maxTokenCount":max_tokens,
                "stopSequences":[],
                "temperature":temperature,
                "topP":top_p
            }
        })
    elif 'anthropic' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "max_tokens_to_sample":max_tokens,
            "temperature":temperature,
            "top_p":top_p
        })
    elif 'ai21' in modelId:
        body = json.dumps({
            "prompt": prompt_data,
            "maxTokens": max_tokens,
            "stopSequences":[],
            "temperature":temperature,
            "topP":top_p
        })
    elif 'stability' in modelId:
        body = json.dumps({
            "text_prompts":[{
                "text":prompt_data
            }]
        }) 
    else:
        print('Parameter model must be one of titan, claude, j2, or sd')
        return

    before = datetime.now()
    response = bedrock_client.invoke_model(body=body, modelId=modelId)
    latency = (datetime.now() - before)
    response_body = json.loads(response.get('body').read())    

    if 'amazon' in modelId:
        response = response_body.get('results')[0].get('outputText')
    elif 'anthropic' in modelId:
        response = response_body.get('completion')
    elif 'ai21' in modelId:
        response = response_body.get('completions')[0].get('data').get('text')

    return response, latency