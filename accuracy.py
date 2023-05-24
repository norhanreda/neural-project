with open('results.txt', 'r') as f1, open('results_set_1.txt', 'r') as f2:
    file1_lines = f1.readlines()
    file2_lines = f2.readlines()
    
    # Find the lines that are different between the two files
    diff_lines = 0
    total_lines = len(file1_lines)
    for i, line in enumerate(file1_lines):
        if line != file2_lines[i]:
            diff_lines += 1
    
    # Calculate the accuracy
    accuracy = (total_lines - diff_lines) / total_lines
    
    # Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))