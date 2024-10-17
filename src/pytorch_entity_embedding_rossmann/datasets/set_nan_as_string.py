def set_nan_as_string(data, replace_str="0"):
    """
    Replaces '' with the '0' string.
    Original code in:
        https://github.com/entron/entity-embedding-rossmann/blob/master/extract_csv_files.py
    """
    for i, x in enumerate(data):
        for key, value in x.items():
            if value == "":
                x[key] = replace_str
        data[i] = x
