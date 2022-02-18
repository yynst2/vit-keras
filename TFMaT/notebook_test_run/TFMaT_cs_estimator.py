import sys,os
import argparse

sys.path.append('/ocean/projects/bio220011p/xcheni')
# cerebras common_zoo
from common_zoo.estimator.tf.run_config import CSRunConfig
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.estimator.utils import (
    cs_disable_summaries,
    cs_enable_summaries,
)
from modelzoo.common.tf.run_utils import (
    check_env,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)
from modelzoo.fc_mnist.tf.data import eval_input_fn, train_input_fn
from modelzoo.fc_mnist.tf.model import model_fn
from modelzoo.fc_mnist.tf.utils import (
    DEFAULT_YAML_PATH,
    get_custom_stack_params,
    get_params,
)
#
import tensorflow as tf
from test.notebook_test_run.TFMaT import *
import tensorflow_addons as tfa
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

# params
# params={}
# params['sequence_length']=1000
# params['motif_size']=6000
# params['motif_length_max']=35
# params['num_layers']=12
# params['num_heads']=12
# params['name']='TFMaT_test'
# params['hidden_size']=768
# params['mlp_dim']=3072
# params['classes']=1000
# params['dropout']=0.1
# params['activation']='sigmoid'
# params['representation_size']=768
# params['motif_embedding_trainable']=False
# params["lr"]=0.001

# model_fn
def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    # x = tf.keras.layers.Input(shape=(params['sequence_length'],4))
    model_params=params['model']
    motif_embedding = tf.keras.layers.Conv1D(
        filters=model_params['motif_size'],
        kernel_size=model_params['motif_length_max'],
        strides=1,
        padding="same",
        name="motif_embedding",
    )
    motif_embedding.trainable = model_params['motif_embedding_trainable']
    y = motif_embedding(features)
    y = tf.keras.layers.Dense(
        units=model_params['hidden_size'],
        name="motif_to_hidden_embedding"
    )(y)
    y = layers.ClassToken(name="class_token")(y)
    y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(model_params['num_layers']):
        y, _ = layers.TransformerBlock(
            num_heads=model_params['num_heads'],
            mlp_dim=model_params['mlp_dim'],
            dropout=model_params['dropout'],
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0],
                               name="ExtractToken")(y)

    y = tf.keras.layers.Dense(
        model_params['representation_size'],
        name="pre_logits",
        activation="tanh"
    )(y)

    logits = tf.keras.layers.Dense(
        model_params['classes'],
        name="head",
        activation=model_params['activation'])(y)

    learning_rate = tf.constant(model_params["lr"])
    loss_op = None
    train_op = None

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss_op = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )
    train_op = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    ).minimize(loss_op, global_step=tf.train.get_global_step())
    spec = CSEstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)


    return spec

# deal with these later after model compiles
def input_fn(params):
    batch_size = 128
    dna_seq = np.random.random(size=(12800, 1, 1000, 4))
    dna_seq_exp = np.expand_dims(dna_seq, axis=-1)
    labels = np.random.randint(low=0, high=1000, size=(12800, 1))
    labels_cat = to_categorical(labels, num_classes=1000)
    labels_cat = np.expand_dims(labels_cat, 1)

    dataset = Dataset.from_tensor_slices((dna_seq, labels_cat))
    dataset.shuffle(128, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    return dataset


# config
config = CSRunConfig(
    cs_ip=ip,
    save_checkpoints_steps=1000,
    log_step_count_steps=10000
)


# estimator
est = CerebrasEstimator(
    model_fn=model_fn,
    config=config,
    params=params,
    model_dir='./out',
)

# train
est.train(input_fn=input_fn, steps=100000, use_cs=True)


def create_arg_parser(default_model_dir):
    """
    Create parser for command line args.

    :param str default_model_dir: default value for the model_dir
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        default=DEFAULT_YAML_PATH,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--cs_ip",
        default=None,
        help="Cerebras System IP address, defaults to None. Ignored on GPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of steps to run mode train."
            + " Runs repeatedly for the specified number."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode train or for defining training"
            + " configuration for train_and_eval. Runs incrementally till"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode eval, eval_all or for defining"
            + " eval configuration for train_and_eval. Runs once for"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["train", "eval", "eval_all", "train_and_eval"],
        help=(
            "Can train, eval, eval_all, or train_and_eval."
            + "  Train and eval will compile and train if on the Cerebras System,"
            + "  and just run locally (CPU/GPU) if not on the Cerebras System."
            + "  train_and_eval will run locally."
            + "  Eval_all will run eval locally for all available checkpoints."
        ),
    )
    parser.add_argument(
        "--multireplica",
        action="store_true",
        help="run multiple copies of the model data-parallel"
        + " on the wafer at the same time.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Compile model up to kernel matching.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force model to run on a specific device (e.g., --device /gpu:0)",
    )

    return parser


def validate_params(params):
    # check validate_only/compile_only
    runconfig_params = params["runconfig"]
    assert not (
        runconfig_params["validate_only"] and runconfig_params["compile_only"]
    ), "Please only use one of validate_only and compile_only."

    # ensure runconfig is compatible with the Cerebras System
    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        assert (
            runconfig_params["mode"] == "train"
        ), "For FC_MNIST model, only training is supported on the Cerebras System."
    else:
        assert not runconfig_params[
            "multireplica"
        ], "Multi-replica training is only possible on the Cerebras System."


def run(
    args, params, model_fn, train_input_fn=None, eval_input_fn=None,):
    """
    Set up estimator and run based on mode

    :params dict params: dict to handle all parameters
    :params tf.estimator.EstimatorSpec model_fn: Model function to run with
    :params tf.data.Dataset train_input_fn: Dataset to train with
    :params tf.data.Dataset eval_input_fn: Dataset to validate against
    """
    # update and validate runtime params
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    validate_params(params)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    stack_params = get_custom_stack_params(params)

    # prep cs1 run environment, run config and estimator
    check_env(runconfig_params)
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"],
        stack_params=stack_params,
        **csrunconfig_dict,
    )
    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
    )

    # execute based on mode
    if runconfig_params["validate_only"] or runconfig_params["compile_only"]:
        if runconfig_params["mode"] == "train":
            input_fn = train_input_fn
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            input_fn = eval_input_fn
            mode = tf.estimator.ModeKeys.EVAL
        est.compile(
            input_fn, validate_only=runconfig_params["validate_only"], mode=mode
        )
    elif runconfig_params["mode"] == "train":
        est.train(
            input_fn=train_input_fn,
            steps=runconfig_params["steps"],
            max_steps=runconfig_params["max_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval":
        est.evaluate(
            input_fn=eval_input_fn,
            steps=runconfig_params["eval_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval_all":
        ckpt_list = tf.train.get_checkpoint_state(
            runconfig_params["model_dir"]
        ).all_model_checkpoint_paths
        for ckpt in ckpt_list:
            est.evaluate(
                input_fn=eval_input_fn,
                checkpoint_path=ckpt,
                steps=runconfig_params["eval_steps"],
                use_cs=use_cs,
            )
    elif runconfig_params["mode"] == "train_and_eval":
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=runconfig_params["max_steps"],
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=runconfig_params["eval_steps"],
            throttle_secs=runconfig_params["throttle_secs"],
        )
        tf.estimator.train_and_evaluate(est, train_spec, eval_spec)


def main():
    """
    Main function
    """
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )
    parser = create_arg_parser(default_model_dir)
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)
    summary_context = (
        cs_disable_summaries if args.multireplica else cs_enable_summaries
    )
    with summary_context():
        run(
            args=args,
            params=params,
            model_fn=model_fn,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
        )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
