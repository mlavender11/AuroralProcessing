import argparse
from . import video_cli, extract_cli, processing_cli, flow_cli

COMMANDS = [video_cli, extract_cli, processing_cli, flow_cli]


def build_parser():
    parser = argparse.ArgumentParser(prog="aurora", description="HiST/DMC auroral processing")
    sub = parser.add_subparsers(title="commands", dest="command", required=True, metavar="COMMAND")
    for module in COMMANDS:
        module.add_parser(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)  # TODO or 0?
