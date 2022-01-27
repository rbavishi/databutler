
def read_file(filename):
    """
    Accepts a file name and returns the contents of that file as a string.
    If file could not be read, raises Exception.

    :param filename The name of the file to be read.
    """
    from os import path
    if  not path.exists(filename) or not path.isfile(filename):
        raise Exception('invalid file path provided')
    with open(filename, 'r') as file:
        file_contents = file.read()
        return file_contents