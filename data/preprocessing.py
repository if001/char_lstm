import sys
from itertools import chain
import os


def first_load(target_file):
    with open(target_file) as f:
        char_list = list(f.read())
    return char_list


def remove_space(char_list):
    char_list = char_list[char_list != " "]
    return char_list


def remove_new_line(char_list):
    char_list = char_list[char_list != "\n"]
    return char_list


def flatten(char_list):
    return list(chain.from_iterable(char_list))


def split_char(flatten_char):
    a = list(flatten_char)
    a = " ".join(a)
    return a


def save_file(char_list, save_file):
    with open(save_file, "w") as f:
        f.write(char_list)
    print("save ", save_file)


def main():
    if len(sys.argv) == 1:
        print("invalid argument")
        exit(0)
    input_file = sys.argv[-1]

    root, ext = os.path.splitext(input_file)

    output_file = root + "-split_space" + ext

    char_list = first_load(input_file)
    # char_list = remove_space(char_lis_lt)
    print("flatten")
    char_list = flatten(char_list)
    print("split")
    char_list = split_char(char_list)
    save_file(char_list, output_file)


if __name__ == "__main__":
    main()
