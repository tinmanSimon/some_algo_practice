#returns number of ways to parenthesizing the s and produce the result
def bool_eval(s, r):
    result = ""
    if r == False: result = "0"
    else: result = "1"
    def helper(s, result, startIndex, endIndex):
        if startIndex >= endIndex: raise ValueError("Error - bool_eval::helper - msg: startIndex >= endIndex")
        if startIndex + 1 == endIndex: 
            if s[startIndex] == result: return 1
            else: return 0
        ret = 0
        for i in range(startIndex + 1, endIndex, 2):
            leftTrue = helper(s, "1", startIndex, i)
            rightTrue = helper(s, "1", i + 1, endIndex)
            leftFalse = helper(s, "0", startIndex, i)
            rightFalse = helper(s, "0", i + 1, endIndex)
            if s[i] == "&": 
                if result == "1": ret += leftTrue * rightTrue
                else: ret += leftTrue * rightFalse + leftFalse * rightTrue + leftFalse * rightFalse
            elif s[i] == "|": 
                if result == "1": ret += leftTrue * rightTrue + leftTrue * rightFalse + leftFalse * rightTrue
                else: ret += leftFalse * rightFalse
            elif s[i] == "^": 
                if result == "1": ret += leftTrue * rightFalse + leftFalse * rightTrue
                else: ret += leftTrue * rightTrue + leftFalse * rightFalse
        return ret
    
    return helper(s, result, 0, len(s))

print(bool_eval("1^1|0&0^1", True))