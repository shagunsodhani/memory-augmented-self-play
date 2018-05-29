import argparse


def str2bool(v):
    # Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argument_parser(config):
    delimiter = "_"
    parser = argparse.ArgumentParser()
    for key1 in config:
        for key2 in config[key1]:
            argument = "--" + key1 + delimiter + key2
            _type = type(config[key1][key2])
            if (_type == bool):
                parser.add_argument(argument, help="Refer config.cfg to know about the config params",
                                    type=str2bool)
            else:
                parser.add_argument(argument, help="Refer config.cfg to know about the config params",
                                    type=_type)
    args = vars(parser.parse_args())

    for key, value in args.items():
        key1, *key2 = key.split(delimiter)
        key2 = delimiter.join(key2)
        if value:
            config[key1][key2] = value

    return config
