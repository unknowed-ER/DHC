import collections
import functools
import os
import pathlib
from datetime import timezone, datetime, timedelta
import time
import sys
import argparse
from collections import namedtuple
from namedlist import namedlist
import pdb
from ruamel.yaml import YAML

__PATH__ = os.path.abspath(os.path.dirname(__file__))
DEFAULT_YML_FNAME = os.path.join(__PATH__, 'ymls/default.yml')


class CommandArgs:
    """Singleton version of collections.defaultdict
    """
    def __new__(cls):
        if not hasattr(cls, 'instance') or not cls.instance:
            cls.instance = collections.defaultdict(list)
        return cls.instance


def add_argument(*args, **kwargs):
    def decorator(func):
        _command_args = CommandArgs()
        _command_args[func.__name__].append((args, kwargs))

        @functools.wraps(func)
        def f(*args, **kwargs):
            ret = func(*args, **kwargs)
            return ret
        return f
    return decorator


def initialize_argparser(commands, command_args,
                         parser_cls=argparse.ArgumentParser):
    ''' commands: model name
        command_args: args from command line
        parser_cls: args from argparser file
    '''
    # Set default arguments
    parser = parser_cls(description=__doc__,
                        formatter_class=argparse.RawTextHelpFormatter)
    if hasattr(parser, "_add_default_arguments"):
        parser._add_default_arguments()
    parser.add_argument("--cfg", type=str, default="ymls/default.yml")
    subparsers = parser.add_subparsers(title='Available models', dest="model")

    # Set model-specific arguments
    sps = {}
    for (cmd, action) in commands.items():
        sp = subparsers.add_parser(cmd, help=action.__doc__)
        for (args, kwargs) in command_args.get(action.__name__, []):
            sp.add_argument(*args, **kwargs)
        sps[cmd] = sp

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    else:
        args = parser.parse_args()
        model_args, _ = sps[args.model].parse_known_args()

    return args, model_args


def jupyter_initialize_argparser(commands, command_args,
                         parser_cls=argparse.ArgumentParser, arg_shell_list=[]):
    ''' commands: model name
        command_args: args from command line
        parser_cls: args from argparser file
    '''
    # Set default arguments
    parser = parser_cls(description=__doc__,
                        formatter_class=argparse.RawTextHelpFormatter)
    if hasattr(parser, "_add_default_arguments"):
        parser._add_default_arguments()
    parser.add_argument("--cfg", type=str, default="ymls/default.yml")
    subparsers = parser.add_subparsers(title='Available models', dest="model")

    # Set model-specific arguments
    sps = {}
    for (cmd, action) in commands.items():
        sp = subparsers.add_parser(cmd, help=action.__doc__)
        for (args, kwargs) in command_args.get(action.__name__, []):
            sp.add_argument(*args, **kwargs)
        sps[cmd] = sp

    # print('commands:',commands)
    # print('sps:',sps)
    args = parser.parse_args(args=arg_shell_list)
    model_args, _ = sps[args.model].parse_known_args(args=[])
    # print('args:', args)
    # print('model_args:', model_args)

    return args, model_args


def create_or_load_hparams(args, model_args, yaml_fname):
    ''' model_args: return by initialize_argparser
        yaml_fname: default.yaml
    '''
    args = vars(args)
    model_args = vars(model_args)

    # HParams = namedtuple('HParams', args.keys())  # args already contain model_args
    HParams = namedlist('HParams', args.keys())  # args already contain model_args
    hparams = HParams(**args)

    # Overwrite params from args (must be predefined in args)
    with open(yaml_fname, 'r') as fp:
        params_from_yaml = YAML().load(fp)
    if 'default' in params_from_yaml:
        for key, value in params_from_yaml['default'].items():
            hparams = hparams._replace(**{key: value})
    if 'model' in params_from_yaml:
        for key, value in params_from_yaml['model'].items():
            hparams = hparams._replace(**{key: value})

    # Set num_gpus, checkpoint_dir
    d = datetime.utcnow().replace(tzinfo=timezone.utc)
    current_time = d.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d-%H-%M-%S')
    # current_time = time.strftime("%Y%m%d%H%M%S")
    print(current_time)
    if hparams.checkpoint_dir == 'unset':
        checkpoint_dir = os.path.join(hparams.checkpoint_base_dir,
                                    hparams.data_name,
                                    hparams.model,
                                    f"{current_time}_{hparams.other_info}")
                                    #   f"save_{hparams.other_info}")
                                    #   f"{current_time}_{hparams.other_info}")
        num_gpus = len(hparams.gpus.split(','))
        hparams = hparams._replace(num_gpus=num_gpus, checkpoint_dir=checkpoint_dir)

        # Save hparams to checkpoint dir (Separate default params and model params)
        model_keys = set(model_args.keys())
        default_keys = set(args.keys()) - model_keys
        dump_yaml_fname = os.path.join(checkpoint_dir, 'params.yml')
        dump_dict = {
            'model': {k: getattr(hparams, k) for k in sorted(model_keys)},
            'default': {k: getattr(hparams, k) for k in sorted(default_keys)},
        }
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(dump_yaml_fname, 'w') as fp:
            YAML().dump(dump_dict, fp)
    else:
        dump_yaml_fname = os.path.join(hparams.checkpoint_dir, 'params.yml')
        with open(dump_yaml_fname, 'r') as fp:
            dump_dict = YAML().load(fp)
        # dump_dict = dict(dump_dict)
        # dump_dict['model'] = dict(dump_dict['model'])
        # dump_dict['default'] = dict(dump_dict['default'])
        hparams = hparams._replace(num_gpus=dump_dict['default']['num_gpus'])

    return hparams, dump_dict


def rewrite_hparams(hparams):
    dump_yaml_fname = os.path.join(hparams.checkpoint_dir, "params.yml")
    with open(dump_yaml_fname) as fp:
        hparams_ori = YAML().load(fp)
    model_keys = tuple(hparams_ori["model"].keys())
    default_keys = tuple(hparams_ori["default"].keys())
    dump_dict = {
        'model': {k: getattr(hparams, k) for k in sorted(model_keys)},
        'default': {k: getattr(hparams, k) for k in sorted(default_keys)},
    }

    pathlib.Path(hparams.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(dump_yaml_fname, 'w') as fp:
        YAML().dump(dump_dict, fp)

def load_hparams(fname):
    with open(fname) as fp:
        hp_dict_ori = YAML().load(fp)
    return hp_dict_ori
