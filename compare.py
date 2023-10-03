import torch
from itertools import cycle, compress
import numpy as np


def _acc(bin1, bin2, bit_redundancy):
        assert bin1.shape == bin2.shape
        N,M = bin1.shape
        total_acc = 0
        interval = bit_redundancy
        ele_num = 8 - bit_redundancy
        
        # Custom slicing in List
        # using compress() + cycle()
        temp = cycle([True] * ele_num + [False] * interval)
        for iii in range(N):
            bits1 = bin1[iii,:]
            # print(len(bits1))
            bits1 = list(compress(bits1.tolist(), temp))
            # print(len(bits1))
            bits2 = bin2[iii,:]
            bits2 = list(compress(bits2.tolist(), temp))
            
            bits1 = np.array(bits1)
            bits2 = np.array(bits2)
            acc = np.equal((bits1 >= 0.5), (bits2 >= 0.5)).sum().astype(np.float32) / bits1.size
            # acc = (bits1 >= 0.5).eq(bits2 >= 0.5).sum().float() / bits1.numel()
            total_acc += acc
        total_acc = total_acc / N
        return total_acc

if __name__ == "__main__":
    acc = 0.0

    # Open the first file for reading
    with open("./embedded_message.txt", 'r') as file1:
        # Open the second file for reading
        with open("./extracted_message.txt", 'r') as file2:
            line_number = 1  # Initialize a counter for the line number
            
            # Iterate through each line in both files simultaneously
            for line1, line2 in zip(file1, file2):
                # Split each line into a list of numbers using commas as the delimiter
                numbers1 = line1.strip().split(',')
                numbers2 = line2.strip().split(',')
                
                # Convert the list of strings to a list of integers
                numbers1 = [int(num) for num in numbers1]
                numbers2 = [int(num) for num in numbers2]

                numbers1 = torch.tensor([numbers1])
                numbers2 = torch.tensor([numbers2])

                new_acc = _acc(numbers1, numbers2, 7)

                print(new_acc)
                if new_acc>0.5:
                    acc += 1
                
                # acc += new_acc
                
                line_number += 1  # Increment the line number

    print("final acc: ", acc/(line_number-1))
