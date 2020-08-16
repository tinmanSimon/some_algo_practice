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

print(n_queens(8))
