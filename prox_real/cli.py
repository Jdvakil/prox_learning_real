"""Shared sub-command dispatch for data.py, model.py and utils.py.

Each command maps to "module:function" inside prox_real; the module is imported only when its
command runs, so e.g. data.py never imports torch. The function gets the remaining arguments.
"""
import importlib
import sys


def run(prog, summary, commands, argv=None):
    """commands: {name: ("module:function", one-line help)}, in the order they are listed."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in commands:
        width = max(map(len, commands))
        lines = [f"usage: python {prog} <command> [options]   (python {prog} <command> -h for its options)",
                 "", summary, "", "commands:"]
        lines += [f"  {name:<{width}}  {help}" for name, (_, help) in commands.items()]
        print("\n".join(lines))
        if argv and argv[0] not in ("-h", "--help"):
            raise SystemExit(f"\nunknown command: {argv[0]}")
        return 0
    name, rest = argv[0], argv[1:]
    module, function = commands[name][0].split(":")
    sys.argv = [f"{prog} {name}", *rest]        # argparse uses this as the program name in --help
    return getattr(importlib.import_module(f"prox_real.{module}"), function)(rest)
