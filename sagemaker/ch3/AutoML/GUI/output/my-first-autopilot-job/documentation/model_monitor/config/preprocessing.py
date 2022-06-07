import csv

EXCEPTION_MSG = "Failed to map {} label. Label is {} but labels length is {}"


def preprocess_handler(csv_line):
    """ This scripts translates ordinal labels to actual label """

    labels = ['no', 'yes']
    labels_count = len(labels)

    reader = csv.reader([csv_line])
    csv_record = next(reader)

    raw_c0 = int(float(csv_record[0]))
    raw_c2 = int(float(csv_record[2]))

    if raw_c0 >= labels_count:
        raise IndexError(
            EXCEPTION_MSG.format("ground truth", raw_c0, labels_count)
        )
    c0 = labels[raw_c0]

    if raw_c2 >= labels_count:
        raise IndexError(
            EXCEPTION_MSG.format("prediction", raw_c2, labels_count)
        )
    c2 = labels[raw_c2]

    return {'_c0': c0, '_c1': csv_record[1], '_c2': c2}
