import os
import argparse

from cowc import Cowc


def parseArguments(args=None):

    """
    parse command line arguments
    """

    # parse configuration
    parser = argparse.ArgumentParser(description='cowc data prep')
    parser.add_argument('data_path', action="store")
    parser.add_argument('out_path', action="store")

    return parser.parse_args(args)


def main():

    """
    main path of execution
    """

    # parse arguments
    args = parseArguments()
    
    # create image and annotations for keras-yolo3
    cowc = Cowc()
    cowc.process( args.data_path, args.out_path )

    return


# execute main
if __name__ == '__main__':
    main()

