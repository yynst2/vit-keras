import os
import yaml

# todo: compatible with TFMaT

_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(_curdir, "configs/params.yaml")

def get_params(params_file=DEFAULT_YAML_PATH):
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def get_custom_stack_params(params):
    stack_params = {}
    if params["runconfig"]["multireplica"]:
        from cerebras.pb.stack.full_pb2 import FullConfig

        config = FullConfig()
        config.target_num_replicas = -1
        stack_params["config"] = config
        os.environ["CEREBRAS_CUSTOM_MONITORED_SESSION"] = "True"
    return stack_params

