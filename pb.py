import os

def scrape_py_files(directory, output_file):
    with open(output_file, 'w') as outfile:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as infile:
                        outfile.write(f"--- {file_path} ---\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")

if __name__ == "__main__":
    directory = "scripts"  # replace with your target directory
    output_file = "output.txt"  # replace with your desired output file
    scrape_py_files(directory, output_file)
