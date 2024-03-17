import os
import shutil
import random


def randomize_files(files, file_n):
    random.seed()
    random_numbers = random.sample(
        range(0, len(files)), file_n if file_n <= len(files) else len(files)
    )
    return [files[n] for n in random_numbers]


def process_directory(path_from: str, path_to: str, file_n):
    if not os.path.exists(path_to):
        os.mkdir(path_to)
    directories = [
        d for d in os.listdir(path_from) if os.path.isdir(path_from + "\\" + d)
    ]
    for directory in directories:
        path_dir_from = path_from + "\\" + directory
        files = [
            f
            for f in os.listdir(path_dir_from)
            if os.path.isfile(path_dir_from + "\\" + f)
        ]
        files = randomize_files(files, file_n)
        path_dir_to = path_to + "\\" + directory
        if not os.path.exists(path_dir_to):
            os.mkdir(path_dir_to)
        for file in files:
            path_file_from = path_dir_from + "\\" + file
            path_file_to = path_to + "\\" + directory + "\\" + file
            shutil.copy(path_file_from, path_file_to)


if __name__ == "__main__":
    base_path = r""  # Base Dataset folder
    training_path = base_path + r"\Training"
    validation_path = base_path + r"\Validation"
    test_path = base_path + r"\Test"
    destination_path = r""  # Destination folder
    destination_path_train = destination_path + r"\Training"
    destination_path_validation = destination_path + r"\Validation"
    destination_path_test = destination_path + r"\Test"
    file_num = 50  # Files to extract from every class
    process_directory(training_path, destination_path_train, file_num)
    process_directory(validation_path, destination_path_validation, file_num)
    process_directory(test_path, destination_path_test, file_num)
