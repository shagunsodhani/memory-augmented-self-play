import argparse
import json

parser = argparse.ArgumentParser(description='Preprocess the log file by filtering non-json line')
parser.add_argument('--input_file_path', type=str,
                    help="Path of the input log file")
parser.add_argument('--output_file_path', type=str,
                    help="Path of the output log file")

args = parser.parse_args()

with open(args.output_file_path, "w") as output_file:
    with open(args.input_file_path, "r") as input_file:
        for log in input_file:
            try:
                data = json.loads(log)
                if (isinstance(data, dict)):
                    output_file.write(log)
            except json.JSONDecodeError as e:
                pass
