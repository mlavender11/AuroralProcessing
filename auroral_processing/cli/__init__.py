import argparse


COMMANDS = [info, convert, clip] 

def build_parser():
    parser = argparse.ArgumentParser(prog='aurora', description="HiST/DMC auroral processing")
    sub = parser.add_subparsers(title='commands', dest='command', required=True, metavar="COMMAND")
    for module in COMMANDS:
        module.add_arser(sub)
    return parser

def main(argv=None):
    args= build_parser().parse_args(argv)
    return args.func(args) or 0