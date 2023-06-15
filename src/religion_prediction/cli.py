from argparse import ArgumentParser

parser = ArgumentParser(prog='Prediction Module')

def main(args=None):
    # print(transliterate_parser.parse_args(args=args))

    args = parser.parse_args(args=args)
    # print('Test')
    print(args)
