from collections import deque
import math 

def isUniqueTwo(s):
    last_c = ""
    for c in sorted(s):
        if last_c == c: return False
        last_c = c
    return True

def test_is_unique():
    s1 = "abcd"
    s2 = "s4fad"
    s3 = ""
    s4 = "23ds2"
    s5 = "hb 627jh=j ()"
    print(isUniqueTwo(s1))
    print(isUniqueTwo(s2))
    print(isUniqueTwo(s3))
    print(isUniqueTwo(s4))
    print(isUniqueTwo(s5))

def checkPermutation(s1, s2):
    print("len s1 = ", len(s1), "len s2 = ", len(s2))
    if len(s1) != len(s2): 
        return False
    ht = {}
    for c in s1: 
        if c in ht:
            ht[c] += 1
        else: 
            ht[c] = 1
    for c in s2:
        if c in ht:
            if ht[c] == 0: return False
            ht[c] -= 1
        else: 
            return False
    return True

def test_is_permutation():
    dataT = (
        ('abcd', 'bacd'),
        ('3563476', '7334566'),
        ('wef34f', 'wffe34'),
    )
    dataF = (
        ('abcd', 'd2cba'),
        ('2354', '1234'),
        ('dcw4f', 'dcw5f'),
    )
    for s1, s2 in dataT:
        print(checkPermutation(s1,s2))
    for s1, s2 in dataF:
        print(checkPermutation(s1,s2))

def urlify(s):
    list_s = s.split()
    result = "%20".join(list_s)
    print(result)


def test_urlify():
    s1 = "askdj sd    klsdaf jiewr       "
    urlify(s1)

def palindrome_permu(s):
    ht = {}
    for c in s:
        c = c.lower()
        if c == ' ': continue
        if c in ht: ht[c] = (ht[c] + 1) % 2
        else: ht[c] = 1
    return sum(ht.values()) < 2

def test_palin_permu():
    data = [
        ('Tact Coa', True),
        ('jhsabckuj ahjsbckj', True),
        ('Able was I ere I saw Elba', True),
        ('So patient a nurse to nurse a patient so', False),
        ('Random Words', False),
        ('Not a Palindrome', False),
        ('no x in nixon', True),
        ('azAZ', True)]

    for [test_string, expected] in data:
        actual = palindrome_permu(test_string)
        print("test_case: ", actual == expected, ", test string: ", test_string)
    
def edit_steps(s1, s2):
    dp = [[-1 for i in range(len(s2))] for j in range(len(s1))]
    def helper(s1, s2, i, j):
        if i == len(s1): return len(s2) - j
        if j == len(s2): return len(s1) - i
        if dp[i][j] != -1: return dp[i][j]
        if s1[i] == s2[j]: return helper(s1, s2, i+1, j+1)
        dp[i][j] = 1 + min(min(helper(s1, s2, i, j+1), helper(s1, s2, i+1, j)), helper(s1, s2, i+1, j+1))
        return dp[i][j]
    steps = helper(s1, s2, 0, 0)
    return steps

def one_edit(s1, s2):
    return edit_steps(s1,s2) < 2

def test_one_edit():
    data = [
        ('pale', 'ple', True),
        ('pales', 'pale', True),
        ('pale', 'bale', True),
        ('paleabc', 'pleabc', True),
        ('pale', 'ble', False),
        ('a', 'b', True),
        ('', 'd', True),
        ('d', 'de', True),
        ('pale', 'pale', True),
        ('pale', 'ple', True),
        ('ple', 'pale', True),
        ('pale', 'bale', True),
        ('pale', 'bake', False),
        ('pale', 'pse', False),
        ('ples', 'pales', True),
        ('pale', 'pas', False),
        ('pas', 'pale', False),
        ('pale', 'pkle', True),
        ('pkle', 'pable', False),
        ('pal', 'palks', False),
        ('palks', 'pal', False)
    ]
    for [test_s1, test_s2, expected] in data:
        actual = one_edit(test_s1, test_s2)
        print("Test result:", actual == expected, ", s1 =", test_s1, ", s2 =", test_s2)

def strComp(s):
    counter = 1
    l = []
    for i in range(1, len(s)):
        if s[i] == s[i-1]: counter += 1
        else: 
            l.append(s[i-1] + str(counter))
            counter = 1
    l.append(s[len(s) - 1] + str(counter))

    return min("".join(l), s, key=len)

def test_strComp():
    data = [
        ('aabcccccaaa', 'a2b1c5a3'),
        ('abcdef', 'abcdef')
    ]

    for [test_string, expected] in data:
        actual = strComp(test_string)
        print("Test result:", actual == expected)

def rotateMat(m):
    def swap(a):
        a[0], a[1] = a[1], a[0]
    mlength = len(m)
    for layer in range(mlength // 2):
        topRow = layer
        leftCol = layer
        bottomRow = mlength - layer - 1
        rightCol = mlength - layer - 1
        for curRow in range(topRow, bottomRow):
            block = curRow - topRow
            m[curRow][leftCol],  m[topRow][rightCol - block] = m[topRow][rightCol - block], m[curRow][leftCol]
            m[curRow][leftCol], m[bottomRow][leftCol + block] = m[bottomRow][leftCol + block], m[curRow][leftCol]
            m[bottomRow - block][rightCol], m[bottomRow][leftCol + block] = m[bottomRow][leftCol + block], m[bottomRow - block][rightCol]
def test_rotateMat():
    data = [
        ([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ], [
            [21, 16, 11, 6, 1],
            [22, 17, 12, 7, 2],
            [23, 18, 13, 8, 3],
            [24, 19, 14, 9, 4],
            [25, 20, 15, 10, 5]
        ])
    ]

    for [test, expect] in data:
        rotateMat(test)
        print("Test result:", test == expect)#, "\n", test, "\n", expect)


def zeroMat(mat):
    m = len(mat)
    n = len(mat[0])
    firstRowShouldBeZero = False
    firstColShouldBeZero = False
    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                mat[0][j] = 0
                mat[i][0] = 0
                if i == 0: firstRowShouldBeZero = True
                if j == 0: firstColShouldBeZero = True
    for i in range(1,m):
        for j in range(1,n):
            if mat[0][j] == 0 or mat[i][0] == 0: mat[i][j] = 0
    if firstRowShouldBeZero: 
        for j in range(n): mat[0][j] = 0
    if firstColShouldBeZero:
        for i in range(m): mat[i][0] = 0

def test_zeroMat():
    data = [
        ([
            [1, 2, 3, 4, 0],
            [6, 0, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 0, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ], [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [11, 0, 13, 14, 0],
            [0, 0, 0, 0, 0],
            [21, 0, 23, 24, 0]
        ])
    ]

    for [test_matrix, expected] in data:
        zeroMat(test_matrix)
        print(test_matrix == expected)
        print(test_matrix)

def strRotate(s1, s2):
    if len(s1) != len(s2): return False
    doubleS1 = s1 + s1
    return s2 in doubleS1

def testStrRotate():
    data = [
        ('waterbottle', 'erbottlewat', True),
        ('foo', 'bar', False),
        ('foo', 'foofoo', False)
    ]

    for [s1, s2, expect] in data:
        print("Test result:", strRotate(s1,s2) == expect)

# linked list
def removeDups(l):
    unorSet = set()
    for e in l:
        if not e in unorSet:
            unorSet.add(e)
    return list(unorSet)

def testRemoveDups():
    l = [1,2,3,2,1]
    l = removeDups(l)
    print(l)

def intToBinary(i):
    s = ""
    while True:
        s = str(i % 2) + s
        i //= 2
        if i == 0: break
    return s

def tmpfunc(n):
    total1, total2 = 0, 0
    for i in range(1,n):
        total1 += i * (0.5) ** (i + 1)
        total2 += 0.5 ** i
    print(total1, total2)

def printTowers(towers):
    counter = 0
    for tower in towers:
        counter += 1
        print("tower", counter, ":", "".join(str(tower)))
    print("\n")

# towers_move_top(towers, 1, 2) will move 1st tower's top to 2nd tower's top.
def towers_move_top(towers, target, dest):
    towers[dest].append(towers[target].pop())
    printTowers(towers)

def tower_of_hanoi(n):
    if n < 1: return
    towers = [[], [], []]
    orderQueue = []
    for i in range(n,0,-1):
        towers[0].append(i)
    print("-------------------\nStart of the tower")
    printTowers(towers)
    print("-------------------")
    while len(towers[0]) > 0:
        new_val = towers[0][-1]
        if len(towers[1]) == 0: 
            towers_move_top(towers, 0, 1)
        else: 
            towers_move_top(towers, 0, 2)
        orderQueue.append(new_val)
        while len(towers[1]) != len(orderQueue) and len(towers[2]) != len(orderQueue):
            cur_node = orderQueue[0]
            if len(towers[1]) > 0 and cur_node == towers[1][-1]:
                if len(towers[2]) > 0 and (cur_node & 1) != (towers[2][-1] & 1) and cur_node < towers[2][-1]:
                    towers_move_top(towers, 1, 2)
                elif len(towers[0]) > 0 and (cur_node & 1) != (towers[0][-1] & 1) and cur_node < towers[0][-1]: 
                    towers_move_top(towers, 1, 0)
                elif len(towers[2]) == 0:
                    towers_move_top(towers, 1, 2)
                elif len(towers[0]) == 0:
                    towers_move_top(towers, 1, 0)
            elif len(towers[2]) > 0 and cur_node == towers[2][-1]:
                if len(towers[1]) > 0 and (cur_node & 1) != (towers[1][-1] & 1) and cur_node < towers[1][-1]: 
                    towers_move_top(towers, 2, 1)
                elif len(towers[0]) > 0 and (cur_node & 1) != (towers[0][-1] & 1) and cur_node < towers[0][-1]: 
                    towers_move_top(towers, 2, 0)
                elif len(towers[1]) == 0:
                    towers_move_top(towers, 2, 1)
                elif len(towers[0]) == 0:
                    towers_move_top(towers, 2, 0)
            elif len(towers[0]) > 0 and cur_node == towers[0][-1]:
                if len(towers[1]) > 0 and (cur_node & 1) != (towers[1][-1] & 1) and cur_node < towers[1][-1]: 
                    towers_move_top(towers, 0, 1)
                elif len(towers[2]) > 0 and (cur_node & 1) != (towers[2][-1] & 1) and cur_node < towers[2][-1]: 
                    towers_move_top(towers, 0, 2)
                elif len(towers[1]) == 0:
                    towers_move_top(towers, 0, 1)
                elif len(towers[2]) == 0:
                    towers_move_top(towers, 0, 2)
            orderQueue.append(orderQueue.pop(0))
    print("-------------------\nEnd of the tower")
    printTowers(towers)
    print("-------------------")
    return



def permu_with_dups(s):
    def permu_with_dups_helper(s):
        ht = set()
        if s in ht: return []
        if len(s) == 0: return []
        if len(s) == 1: return [s]
        res = []
        for i in range(len(s)):
            list_of_strings = permu_with_dups_helper(s[:i] + s[i+1:])
            for j in list_of_strings:
                new_s = s[i] + j
                if not new_s in ht: 
                    res.append(new_s)
                    ht.add(new_s)
        return res

    res = permu_with_dups_helper(s)
    print(res)
    

def paren(n):
    ht, ht["("], ht[")"], paren_stack, valid_check_stack, res = {}, n, n, ["" for i in range(2*n)], [], []
    def paren_helper(paren_stack_index):
        if ht["("] == 0 and ht[")"] == 0: 
            res.append("".join(paren_stack))
            paren_stack_index -= 1
            return
        if ht["("] > 0:
            ht["("] -= 1
            paren_stack[paren_stack_index] = "("
            valid_check_stack.append("(")
            paren_helper(paren_stack_index + 1)
            valid_check_stack.pop()
            ht["("] += 1
        if ht[")"] > 0 and len(valid_check_stack) > 0 and valid_check_stack[-1] == "(":
            ht[")"] -= 1
            valid_check_stack.pop()
            paren_stack[paren_stack_index] = ")"
            paren_helper(paren_stack_index + 1)
            valid_check_stack.append("(")
            ht[")"] += 1
        return
    paren_helper(0)
    return res

def coins(n, coins_values = [25, 10, 5, 1]):
    dp = {}

    def coins_helper(k, highest_change):
        if k == 0: 
            return [""]
        if k in dp and highest_change in dp[k]:
            if k >= 5 and k % 5 == 0: 
                t = k
            return dp[k][highest_change]
        res = []
        for coin_val in coins_values:
            if k >= coin_val and highest_change >= coin_val:
                ret = coins_helper(k - coin_val, coin_val)
                for coinstr in ret:
                    res.append(str(coin_val) + " " + coinstr)
        if not k in dp: dp[k] = {}
        dp[k][highest_change] = res
        return res
    result = coins_helper(n, coins_values[0])
    return result

#prints the all possible ways to put n queens on a board.
#returns the positions of queens for all possible solutions.
def n_queens(n, board_size = 8):
    if n > board_size or n == 0: return []
    board = [[0 for j in range(board_size)] for i in range(board_size)]
    queen_positions = []
    queen_record = []
    
    #returns true if board[i][j] is available for a new queen
    def isValid(i, j):
        for [pos_x, pos_y] in queen_positions:
            if pos_x == i or pos_y == j or abs(pos_y - j) == abs(pos_x - i):
                return False
        return True

    def printBoard():
        board = [[0 for j in range(board_size)] for i in range(board_size)]
        for [pos_x, pos_y] in queen_positions:
            board[pos_x][pos_y] = 1
        for lst in board:
            print(lst)

    def helper(x, left_queens):
        if left_queens == 0:
            queen_record.append(queen_positions[:])
            printBoard()
            print("*******************************\n")
            return
        for i in range(x, board_size):
            for j in range(0, board_size):
                if isValid(i, j):
                    queen_positions.append([i,j])
                    helper(i, left_queens - 1)
                    queen_positions.pop()
        return
    helper(0,n)
    return queen_record

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
