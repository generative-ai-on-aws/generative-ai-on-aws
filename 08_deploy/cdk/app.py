# v2
import aws_cdk as cdk
from cdk.cdk_stack import CdkStack
from constructs import Construct
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_applicationautoscaling as autoscaling
from aws_cdk import App, Stack

# USAGE
# cdk deploy \
#   --parameters model="distilbert-base-uncased-finetuned-sst-2-english" \
#   --parameters task="text-classification" \
#   --parameters insatnce_type="ml.m5.xlarge"
#
# There might be an issue when using
#   cdk deploy --parameters ...
# as documented here
#   https://github.com/aws/aws-cdk/issues/6119
# Hard-coded the params for now

LATEST_PYTORCH_VERSION = "1.8.1"
LATEST_TRANSFORMERS_VERSION = "4.10.2"

region_dict = {
    "af-south-1": "626614931356",
    "ap-east-1": "871362719292",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
    "ap-northeast-3": "364406365360",
    "ap-south-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ca-central-1": "763104351884",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-north-1": "763104351884",
    "eu-south-1": "692866216735",
    "eu-west-1": "763104351884",
    "eu-west-2": "763104351884",
    "eu-west-3": "763104351884",
    "me-south-1": "217643126080",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}

# policies based on https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html#sagemaker-roles-createmodel-perms
iam_sagemaker_actions = [
    "sagemaker:*",
    "ecr:GetDownloadUrlForLayer",
    "ecr:BatchGetImage",
    "ecr:BatchCheckLayerAvailability",
    "ecr:GetAuthorizationToken",
    "cloudwatch:PutMetricData",
    "cloudwatch:GetMetricData",
    "cloudwatch:GetMetricStatistics",
    "cloudwatch:ListMetrics",
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:DescribeLogStreams",
    "logs:PutLogEvents",
    "logs:GetLogEvents",
    "s3:CreateBucket",
    "s3:ListBucket",
    "s3:GetBucketLocation",
    "s3:GetObject",
    "s3:PutObject",
]


def get_image_uri(
    region="us-east-1",
    transformmers_version=LATEST_TRANSFORMERS_VERSION,
    pytorch_version=LATEST_PYTORCH_VERSION,
    use_gpu=False,
):
    repository = f"{region_dict[region]}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference"
    tag = f"{pytorch_version}-transformers{transformmers_version}-{'gpu-py36-cu111' if use_gpu == True else 'cpu-py36'}-ubuntu18.04"
    return f"{repository}:{tag}"


def is_gpu_instance(instance_type):
    return True if instance_type.split(".")[1][0].lower() in ["p", "g"] else False


class HuggingFaceSagemaker(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Hugging Face Model
        huggingface_model = cdk.CfnParameter(
            self,
            "model",
            type="String",
            default="distilbert-base-uncased-finetuned-sst-2-english",
        ).value_as_string

        # Model Task
        huggingface_task = cdk.CfnParameter(
            self,
            "task",
            type="String",
            default="text-classification",
        ).value_as_string

        # Execution role for SageMaker, will be created if not provided
        instance_type = cdk.CfnParameter(
            self,
            "instance_type",
            type="String",
            default="ml.m5.xlarge",
        ).value_as_string

        # Execution role for SageMaker, will be created if not provided
        # we cannot use `CfnParameter` since the value is a `TOKEN` when synthizing the stack
        execution_role = kwargs.pop("role", None)

        # creates the image_uir based on the instance type and region
        use_gpu = is_gpu_instance(instance_type)

        image_uri = get_image_uri(#region=self.region, 
                                  use_gpu=use_gpu)

        # creates new iam role for sagemaker using `iam_sagemaker_actions` as permissions or uses provided arn
        if execution_role is None:
            execution_role = iam.Role(
                self, "hf_sagemaker_execution_role", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
            )
            execution_role.add_to_policy(iam.PolicyStatement(resources=["*"], actions=iam_sagemaker_actions))
            execution_role_arn = execution_role.role_arn
        else:
            execution_role_arn = execution_role

        # defines and creates container configuration for deployment
        container_environment = {"HF_MODEL_ID": huggingface_model, "HF_TASK": huggingface_task}
        container = sagemaker.CfnModel.ContainerDefinitionProperty(environment=container_environment, image=image_uri)

        # creates SageMaker Model Instance
        model = sagemaker.CfnModel(
            self,
            "hf_model",
            execution_role_arn=execution_role_arn,
            primary_container=container,
            model_name=f'model-{huggingface_model.replace("_","-").replace("/","--")}',
        )
        model.node.add_dependency(execution_role)

        # Creates SageMaker Endpoint configurations
        endpoint_configuration = sagemaker.CfnEndpointConfig(
            self,
            "hf_endpoint_config",
            endpoint_config_name=f'config-{huggingface_model.replace("_","-").replace("/","--")}',
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_instance_count=1,
                    instance_type=instance_type,
                    model_name=model.model_name,
                    initial_variant_weight=1.0,
                    variant_name=model.model_name,
                )
            ],
        )
        endpoint_configuration.node.add_dependency(model)

        # Creates Real-Time Endpoint
        endpoint = sagemaker.CfnEndpoint(
            self,
            "hf_endpoint",
            endpoint_name=f'endpoint-{huggingface_model.replace("_","-").replace("/","--")}',
            endpoint_config_name=endpoint_configuration.endpoint_config_name,
        )
        endpoint.node.add_dependency(endpoint_configuration)

        target = autoscaling.ScalableTarget(
            self,
            "ScalableTarget",
            service_namespace=autoscaling.ServiceNamespace.SAGEMAKER,
            scalable_dimension="sagemaker:variant:DesiredInstanceCount",
            min_capacity=1,
            max_capacity=2,
            resource_id=f'endpoint/endpoint-{huggingface_model.replace("_","-").replace("/","--")}/variant/model-{huggingface_model.replace("_","-").replace("/","--")}'
        )
        target.scale_to_track_metric(
            "ScaleToTrackMetric",
            target_value = 500,
            scale_in_cooldown= cdk.Duration.seconds(1500),
            scale_out_cooldown= cdk.Duration.seconds(1500),
            predefined_metric=autoscaling.PredefinedMetric.SAGEMAKER_VARIANT_INVOCATIONS_PER_INSTANCE
        )
        target.node.add_dependency(endpoint)


app = App()

HuggingFaceSagemaker(app, "sagemaker-endpoint"
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.
    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */
    #env=cdk.Environment(account='123456789012', region='us-east-1'),

    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
)

app.synth()
