import argparse

# from https://stackoverflow.com/questions/9234258/in-python-argparse-is-it-possible-to-have-paired-no-something-something-arg
class YesNoAction(argparse.Action):
    '''
    An action that provides a toggle which can be enabled or disabled.

    It can be used like

        parser.add_argument('--foo', action=YesNoAction)

    This generates both a --foo and a --no-foo flag.
    This is useful when one wants to provide various sets of options programmatically
    and needs a way to override the --foo flag.

    >> parser.parse_args(['--foo'])
    Namespace(foo=True)
    >> parser.parse_args(['--no-foo'])
    Namespace(foo=False)
    >> parser.parse_args([])
    Namespace(foo=False)
    '''
    def __init__(self, option_strings, dest, default=False, required=False, help=None):

        if len(option_strings)!=1:
            raise ValueError('Only single argument is allowed with YesNo action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('Yes/No arguments must be prefixed with --')

        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt]
        super().__init__(opts, dest, nargs=0, const=None, default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)