from argparse import ArgumentParser
import os
from random import randint
import shutil


def get_random(files_count, used):
    while True:
        random_number = randint(0, files_count-1)
        if random_number not in used:
            used.append(random_number)
            break
    # print(random_number)
    return random_number


def main():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--data", required=True,
                        help="data to split")
    parser.add_argument("--split", required=True,
                        help="prefix to split by")
    args = parser.parse_args()
    folder = args.data
    prefix = args.split

    if not os.path.exists(os.path.join(folder, 'train')):
        os.mkdir(os.path.join(args.data, 'train'))
        os.mkdir(os.path.join(args.data, 'validation'))
        os.mkdir(os.path.join(args.data, 'test'))

    root = ''
    used = []
    files_list = []
    print(folder)
    path = os.walk(folder, topdown=False)

    for root, dirs, files in path:
        root = root
        for name in files:
            if name.startswith(prefix):
                files_list.append((os.path.join(root, name)))

    files_count = len(files_list)
    training_size = int(files_count * 0.6)
    test_size = int(files_count * 0.2)
    validation_size = int(files_count - training_size - test_size)

    for i in range(training_size):
        rnd = get_random(files_count, used)
        file_name = os.path.join(root, files_list[rnd])
        dst = os.path.join(root, 'train', file_name.split('/')[-1])
        shutil.move(file_name, dst)

    for i in range(validation_size):
        rnd = get_random(files_count, used)
        file_name = os.path.join(root, files_list[rnd])
        dst = os.path.join(root, 'validation', file_name.split('/')[-1])
        shutil.move(file_name, dst)

    for i in range(test_size):
        rnd = get_random(files_count, used)
        file_name = os.path.join(root, files_list[rnd])
        dst = os.path.join(root, 'test', file_name.split('/')[-1])
        shutil.move(file_name, dst)


if __name__ == "__main__":
    main()
