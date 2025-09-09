# utils.py
import orjson


def load_data(input_file):
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(orjson.loads(line))
    return data


def save_data(data, output_file):
    with open(output_file, "wb") as f:  # orjson requires binary mode
        for item in data:
            f.write(orjson.dumps(item) + b"\n")
