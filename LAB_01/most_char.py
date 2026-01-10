#q4

#function to find most occurring character
def most_char(s):
    s = s.lower()          
    max_alph = ''
    max_count = 0

    for alph in s:
        #making sure character is alphabet only
        if alph.isalpha():    
            alph_count = s.count(alph)
            if alph_count > max_count:
                max_count = alph_count
                max_alph = alph

    return max_alph, max_count

#user input
s = input("Enter string: ")
alph, freq = most_char(s)

#output
print("Maximally Occurring Character:", alph)
print("Occurrence count:", freq)
