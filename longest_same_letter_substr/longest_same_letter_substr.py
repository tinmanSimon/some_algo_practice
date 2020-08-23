#returns the longest substring with same letter and no more than k replacements.
#returns the actual replacement times too, like how many times we replace letters.
#Example: longest_same_letter_substr("aaabbbaaaabbaa", 2) -> ["aaaabbaa", 2]
def longest_same_letter_substr(s, k):
    if len(s) < 1: return 0
    ht = {}
    str_start, max_start, max_end, used_k = 0, 0, 0, 0
    target_c = ''
    for str_end in range(len(s)):
        c = s[str_end]
        if not c in ht: ht[c] = 1
        else: ht[c] += 1
        if target_c == '' or ht[c] > ht[target_c]:
            target_c = c

        if(str_end + 1 - str_start - ht[target_c] > k):
            ht[s[str_start]] -= 1
            str_start += 1

        elif max_end - max_start < str_end - str_start:
            max_start, max_end = str_start, str_end
            used_k = str_end + 1 - str_start - ht[target_c]

    return [s[max_start : max_end + 1], used_k]
