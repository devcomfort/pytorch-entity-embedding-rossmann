def csv2dicts(csvfile):
    """
    Reads csv and store each line as json in
    the format:
        ---------
        csv file:
        ---------
        colname1, colname2
        val1, val2
        val3, val4
        .
        .
        .

        -------
        Output:
        -------
        {'colname1': val1, 'colname2': val2}
        {'colname1': val3, 'colname2': val4}
        .
        .
        .

    """
    data = []
    keys = []
    for row_index, row in enumerate(csvfile):
        if row_index == 0:
            keys = row
            continue

        data.append({key: value for key, value in zip(keys, row)})

    return data
