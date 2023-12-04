# Porter Johnson and Will Corbett

import numpy as np

# Calculate edit distance between two words using Global Sequence Alignment algorithm
def edit_distance(word1, word2):
    w1_len = len(word1)
    w2_len = len(word2)
    
    matrix = np.zeros((w1_len + 1, w2_len + 1), dtype = int)
    
    for i in range(0, w1_len + 1):
        matrix[i][0] = i
    for x in range(0, w2_len + 1):
        matrix[0][x] = x
        
    for i in range(1, w1_len + 1):
        for j in range(1, w2_len + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,  
                           matrix[i][j - 1] + 1, 
                           matrix[i - 1][j - 1] + cost) 
        
    return matrix[w1_len][w2_len]

# Input a mispelled word, calculate 5 closest edit distance words from dictionary
def spellSuggestions(word):
    
    f = open("Dictionary.txt")
    
    minDist = []
    suggestions = []
    
    for line in f:
        dictword = line.split("\n")
        distance = edit_distance(word, dictword[0])
        minDist.append((dictword[0], distance))
        
    minDist.sort(key=lambda x: x[1])
        
    for i in range(min(5, len(minDist))):
        suggestions.append(minDist[i][0])
            
    return suggestions
    
misspelledWord = input("Enter a misspelled word: ")
suggestions = spellSuggestions(misspelledWord)
print(f"For the word: {misspelledWord}. The 5 best spelling corrections are: {suggestions}")
