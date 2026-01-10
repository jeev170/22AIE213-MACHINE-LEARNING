def highest_char(s):
    max_char = s[0]
    max_count = 0

    for ch in s:
        count = s.count(ch)
        if count > max_count:
            max_count = count
            max_char = ch

    return max_char, max_count


s = input("Enter string: ")
char, count = highest_char(s)
print("Character:", char)
print("Count:", count)

