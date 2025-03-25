import matplotlib.pyplot as plt

def read_numbers_from_file(filename):
    """Reads numbers from a file, assuming each line contains one number."""
    with open(filename, 'r') as file:
        return [float(line.strip()) for line in file if line.strip().isdigit()]

def plot_histogram(data, bins=50):
    """Plots a histogram from the given data."""
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('# of Hits')
    plt.ylabel('Frequency')
    plt.title('Histogram from File Data')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    filename = 'iter_1'  # Change this to your actual filename
    data = read_numbers_from_file(filename)
    if data:
        plot_histogram(data)
    else:
        print("No valid numbers found in the file.")

if __name__ == "__main__":
    main()
