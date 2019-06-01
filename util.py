from datetime import datetime


def format_filename(model_name, arguments, add_date=True):
    file_name = "_".join(map(str, [model_name] + arguments))
    if add_date:
        file_name += datetime.now().strftime("_%b_%-d")

    return file_name
