import copy
import numpy as np
import sys

# print to stderr for debugging purposes
# remove all debugging statements before submitting your code
#msg = "Given board " + sys.argv[1] + "\n";


#parse the input string, i.e., argv[1]
 
#perform intelligent search to determine the next move

#print to stdout for AtroposGame

# As you can see Zook's algorithm is not very intelligent. He 
# will be disqualified.


#fi = open("error.txt", "w")






def stringToBoard(string):
    """
    input: string, | ex: '[13][302][1003][30002][100003][3000002][121212]LastPlay:null'
    output: int list, | ex: [13][302][1003][30002][100003][3000002][121212]
    
    Cuts out the current state of the board from the std input string
    """
    
    boardString = string.split("L")
    cutBoard = boardString[0].replace("[", "").split("]")
    stringBoard = cutBoard[:-1:]
    charBoard = map(list, stringBoard)
    board = [[int(c) for c in ar] for ar in charBoard]
    return board
    

def playToLast(pos):
    """
    input: int list, | ex: (1, 2, 3, 4)
    output: int list, | ex: [6,3]
    
    Takes the std input coordinate system, changes it to my own coordinate system, and finally changes it to an array index
    """
    # pos = [color, height, left, right]
    size = (pos[1] + pos[2] + pos[3]) - 2
    return ind(pos[2], pos[3], pos[1], size)
    

def stringToPlay(string):
    """
    input: string, | ex: '[13][302][1003][31002][100003][3000002][121212]LastPlay:(1,3,1,3)'
    output: None or int list, | ex: [1, 3, 1, 3]
    
    Takes a string in the std input format, and cuts out the LastPlay
    If the LastPlay is null, None is returned
    If not, the characters for the numbers will be cast to ints and put in a list
    """
    
    boardString = string.split(":")
    cutBoard = boardString[1].replace("(", "").replace(")", "").split(",")
    if cutBoard[0] == 'null':
        return None
    else: 
        return [int(c) for c in cutBoard]
    

def isSpace(pos, board):
    """"
    input: (int list, int list), | ex: ([2, 3], [13][302][1003][31002][100003][3000002][121212]) 
    output: None or boolean | ex: True
    
    First, calculates the coordinate position of an array index for the board
    Then calculates the surrounding coordinates
    Then calculates the indexes of each coordinate
    Then uses the indexes to check if there are empty positions arround the current position
    """
    
    if pos is None:
        return False
    
    cord = coord(pos, board)
    maxSide = len(board) - 2
                 
    upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
    upRight = (cord[0], cord[1] - 1, cord[2] + 1)
    left = (cord[0] - 1, cord[1] + 1, cord[2])
    right = (cord[0] + 1, cord[1] - 1, cord[2])
    downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
    downRight = (cord[0] + 1, cord[1], cord[2] - 1)
    
    upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
    upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
    le = ind(left[0], left[1], left[2], maxSide)
    ri = ind(right[0], right[1], right[2], maxSide)
    downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
    downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
    #print(pos,upL,upR,le,ri,downL,downR)
    if (board[upL[0]][upL[1]] != 0) and \
       (board[upR[0]][upR[1]] != 0) and \
       (board[le[0]][le[1]] != 0) and \
       (board[ri[0]][ri[1]] != 0) and \
       (board[downL[0]][downL[1]] != 0) and \
       (board[downR[0]][downR[1]] != 0):
        return False
    else: 
        return True


def ind(left, right, height, maxSide):
    """
    input: (int, int, int, int) | ex: (1,2,3,4)
    output: int list | ex: [2,1]
    
    Changes from the coordinate system to an array index of the board position
    """
    
    if height == 0:
        return [maxSide+1, left - 1]
    else:
        return [maxSide - height + 1, left]

def coord(ind, board):
    """
    input: (int list, int list) | ex: ([2,1], [13][302][1003][31002][100003][3000002][121212])
    output: (int, int, int) | ex: (1, 2, 4)
    
    Takes in an index for the board and the board
    Will return a coordinate in (left, right, height) order
    """
    
    siz = len(board)
    right = len(board[ind[0]]) -1
    if ind[0] == siz - 1:
        return (ind[1] + 1, right - ind[1] + 1, 0)
    else:
        #if len(ind) < 2: sys.stderr.write(str(ind))
        return (ind[1], right - ind[1], (siz - 1) - ind[0])
    


                    
    

def loser(pos, color, boa):
    """
    input: (int list, int, int list) | ex: ([1, 1], 1, [13][302][1003][30002][100003][3000002][121212])
    output: bolean | ex: False
    
    Finds the coordinate of the position
    Then finds the coordinates around the position
    Then finds the indexes for those positions
    Then finds if there are any 3 color triangles
    """
    # The following code is repeated multiple times in different functions. It all serves the same purpose
    
    board = copy.deepcopy(boa)  
    
    # find the coordinate of the current position and side length
    cord = coord(pos, board)
    maxSide = len(board) - 2
             
    # place a simulated circle on the board
    board[pos[0]][pos[1]] = color
    #print(board, boa)      
    
    # find the coordinates for the surrounding positions
    upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
    upRight = (cord[0], cord[1] - 1, cord[2] + 1)
    left = (cord[0] - 1, cord[1] + 1, cord[2])
    right = (cord[0] + 1, cord[1] - 1, cord[2])
    downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
    downRight = (cord[0] + 1, cord[1], cord[2] - 1)
    
    # find the indexes of the coordinates
    upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
    upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
    le = ind(left[0], left[1], left[2], maxSide)
    ri = ind(right[0], right[1], right[2], maxSide)
    downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
    downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
    
    # check all 6 triangles the current node is part of
    if board[pos[0]][pos[1]] == 0:
        return False
    elif (board[upL[0]][upL[1]] != board[upR[0]][upR[1]]) and (board[upL[0]][upL[1]] != board[pos[0]][pos[1]]) and (board[pos[0]][pos[1]] != board[upR[0]][upR[1]]) and \
         (board[pos[0]][pos[1]] != 0) and (board[upL[0]][upL[1]] != 0) and (board[upR[0]][upR[1]] != 0):
        return True
    elif (board[upL[0]][upL[1]] != board[le[0]][le[1]]) and (board[upL[0]][upL[1]] != board[pos[0]][pos[1]]) and (board[pos[0]][pos[1]] != board[le[0]][le[1]]) and \
         (board[pos[0]][pos[1]] != 0) and (board[upL[0]][upL[1]] != 0) and (board[le[0]][le[1]] != 0):
        return True
    elif (board[downL[0]][downL[1]] != board[le[0]][le[1]]) and (board[downL[0]][downL[1]] != board[pos[0]][pos[1]]) and (board[pos[0]][pos[1]] != board[le[0]][le[1]]) and \
         (board[pos[0]][pos[1]] != 0) and (board[downL[0]][downL[1]] != 0) and (board[le[0]][le[1]] != 0):
        return True
    elif (board[downL[0]][downL[1]] != board[downR[0]][downR[1]]) and (board[downL[0]][downL[1]] != board[pos[0]][pos[1]]) and (board[pos[0]][pos[1]] != board[downR[0]][downR[1]]) and \
         (board[pos[0]][pos[1]] != 0) and (board[downL[0]][downL[1]] != 0) and (board[downR[0]][downR[1]] != 0):
        return True
    elif (board[ri[0]][ri[1]] != board[downR[0]][downR[1]]) and (board[ri[0]][ri[1]] != board[pos[0]][pos[1]]) and (board[pos[0]][pos[1]] != board[downR[0]][downR[1]]) and \
         (board[pos[0]][pos[1]] != 0) and (board[ri[0]][ri[1]] != 0) and (board[downR[0]][downR[1]] != 0):
        return True
    elif (board[ri[0]][ri[1]] != board[upR[0]][upR[1]]) and (board[ri[0]][ri[1]] != board[pos[0]][pos[1]]) and (board[pos[0]][pos[1]] != board[upR[0]][upR[1]]) and \
         (board[pos[0]][pos[1]] != 0) and (board[ri[0]][ri[1]] != 0) and (board[upR[0]][upR[1]] != 0):
        return True
    else:
        return False
        

    
    
def tryAgain(board, last, scores, lookahead, alpha, beta, maximizingPlayer):
    """
    input: (int list, None or int list, int list, int, int, int, boolean) | ex: ([13][302][1003][31002][100003][3000002][121212], None, [[0],[0,0],[0,0,0],[0,0,0,0],[0,0,0,0,0]], 3, float("-inf"), float("inf"), True)
    output: (int, int list) | ex: (-30, [1, 2, 1])
    
    Runs alpha beta reduction on a minimax scoring system
    """
    
    # Base Case
    if lookahead == 0:
        # Base Value for nodes
        return boardScore(last, board)
    else:
        # Alpha Beta Pruning Start
        # ex is for exiting multiple for loops at the same time
        ex = False

        if maximizingPlayer:
            # First check if there has been a play yet and if there is room around that play
            if (last is not None) and (isSpace(last, board)): 
                
                # set temporary value to lowest value possible 
                v = float("-inf")
                # set temporary move to return. If no valid move is found, will use the random player 
                move = [1, last[0], last[1]]
                #######################################################
                # Finds coordinates and indexes for the positions surrounding the last play        
                        
                cord = coord(last, board)
                maxSide = len(board) - 2
                             
                upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
                upRight = (cord[0], cord[1] - 1, cord[2] + 1)
                left = (cord[0] - 1, cord[1] + 1, cord[2])
                right = (cord[0] + 1, cord[1] - 1, cord[2])
                downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
                downRight = (cord[0] + 1, cord[1], cord[2] - 1)
                
                upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
                upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
                le = ind(left[0], left[1], left[2], maxSide)
                ri = ind(right[0], right[1], right[2], maxSide)
                downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
                downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
                
                # Creates a list of all empty valid positions
                neigh = [upL, upR, le, ri, downL, downR]
                emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
                
                
                #######################################################
                # iterate over the valid moves for each color
                for color in range(3):
                    for item in emptyN:
                        # Check the current move won't lose. 
                        # NOTE: If there are only losing moves left, the function will return a non valid move and let the random player lose for it
                        if not loser(item, color+1, board):
                            
                            # Place a token at the current position, recursively call the function with a shortenned lookahead, remove the token
                            la = [item[0], item[1]]
                            board[item[0]][item[1]] = color + 1
                            temp = tryAgain(board, la, scores, lookahead - 1, alpha, beta, False)
                            board[item[0]][item[1]] = 0
                            
                            # Find if the max value changed     
                            v = max(v, temp[0])
                            if v > alpha:
                                # If this is the best move, record it
                                alpha = v
                                move = [color+1, item[0], item[1]]
                                
                            if beta <= alpha:
                                # Break out of the for loops
                                ex = True
                                break
                    if ex:
                        ex = False
                        break
            else:
                # set temporary value to lowest value possible 
                v = float("-inf")
                # set temporary move to return. If no valid move is found, will use the random player 
                move = [1,0,0]
                
                # If there is no last play or no space, check every position
                for c in range(3):
                    for a in range(len(scores[c])):
                        for val in range(len(scores[c][a])):
                            # Checks if the move is valid and won't lose
                            if board[a+1][val+1] != 0 or loser([a+1,val+1], c+1, board):
                                continue
                            else:
                                # Place a token at the current position, recursively call the function with a shortenned lookahead, remove the token
                                board[a+1][val+1] = c + 1
                                la = [a+1,val+1]
                                temp = tryAgain(board, la, scores, lookahead - 1, alpha, beta, False)    
                                board[a+1][val+1] = 0
                                
                                # Find if the max value changed  
                                v = max(v, temp[0])
                                if v > alpha:
                                    # If this is the best move, record it
                                    alpha = v
                                    move = [c+1, a+1, val+1]
                                if beta <= alpha:
                                    # Break out of the for loops
                                    ex = True
                                    break
                        if ex:  
                            break
                    if ex:
                        ex = False
                        break
            # Return the value and the move position
            return v, move
        else:
            # First check if there has been a play yet and if there is room around that play
            if (last is not None) and (isSpace(last, board)): 
                # set temporary value to highest value possible 
                v = float("inf")
                # set temporary move to return. If no valid move is found, will use the random player 
                move = [1, last[0], last[1]]
                #######################################################
                # Finds coordinates and indexes for the positions surrounding the last play          
                        
                cord = coord(last, board)
                maxSide = len(board) - 2
                             
                upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
                upRight = (cord[0], cord[1] - 1, cord[2] + 1)
                left = (cord[0] - 1, cord[1] + 1, cord[2])
                right = (cord[0] + 1, cord[1] - 1, cord[2])
                downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
                downRight = (cord[0] + 1, cord[1], cord[2] - 1)
                
                upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
                upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
                le = ind(left[0], left[1], left[2], maxSide)
                ri = ind(right[0], right[1], right[2], maxSide)
                downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
                downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
                
                # Creates a list of all empty valid positions
                neigh = [upL, upR, le, ri, downL, downR]
                emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
                
                
                #######################################################
                # iterate over the valid moves for each color
                for color in range(3):
                    for item in emptyN:
                        # Check the current move won't lose. 
                        # NOTE: If there are only losing moves left, the function will return a non valid move and let the random player lose for it
                        if not loser(item, color+1, board):
                            # Place a token at the current position, recursively call the function with a shortenned lookahead, remove the token
                            la = [item[0], item[1]]
                            board[item[0]][item[1]] = color + 1
                            temp = tryAgain(board, la, scores, lookahead - 1, alpha, beta, True)
                            board[item[0]][item[1]] = 0
                            
                            # Find if the min value changed  
                            v = min(v, temp[0])
                            if v < beta:
                                beta = v
                                move = [color+1, item[0], item[1]]
                            if beta <= alpha:
                                # Break out of the for loops
                                ex = True
                                break
                    if ex:
                        ex = False
                        break
#                if ([a+1, val+1] not in emptyN) or board[a+1][val+1] != 0:
#                    #print([a+1, val+1])
#                    continue
                
                
                ######################################################
            else:
                v = float("inf")
                move = [1,0,0]
                for c in range(3):
                    for a in range(len(scores[c])):
                        for val in range(len(scores[c][a])):
                            if board[a+1][val+1] != 0 or loser([a+1,val+1], c+1, board):
                                continue
                            else:
                                board[a+1][val+1] = c + 1
                                la = [a+1,val+1]
                                
                                #print("anything")
                                temp = tryAgain(board, la, scores, lookahead - 1, alpha, beta, True)
                                #print temp
                                board[a+1][val+1] = 0
                                #if lookahead == 1: print("max: " + str(temp))
                                v = min(v, temp[0])
                                #if v == 0: fi.write("max: " + str(lookahead) + "\n")
                                #fi.write("max: " + str(v) + "\n")
                                #alpha = max(alpha, v)
                                if v < beta:
                                    # If this is the best move, record it
                                    beta = v
                                    move = [c+1, a+1, val+1]
                                if beta <= alpha:
                                    #print("max: " + str(alpha) + ", " + str(beta) )
                                    ex = True
                                    break
                        if ex:  
                            break
                    if ex:
                        ex = False
                        break
            # return value and move
            return v, move


def boardScore(last, board):
    """
    input: (int list, int list) | ex: ([3,1], [3,1], [[13],[302],[1003],[31002],[100003],[3000002])
    output: int | ex: -30
    
    Calculates the amount of losing spaces around the last play
    Uses this metric to evaluate board values
    """
    
    # Initial score is started at 0
    score = 0
    move = last
    
    # Check if there was a previous move with open neighbors 
    if (last is not None) and (isSpace(last, board)):
        # Finds the coordinates and indexes of the surrounding positions to the last move
        
        cord = coord(last, board)
        maxSide = len(board) - 2
                     
        upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
        upRight = (cord[0], cord[1] - 1, cord[2] + 1)
        left = (cord[0] - 1, cord[1] + 1, cord[2])
        right = (cord[0] + 1, cord[1] - 1, cord[2])
        downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
        downRight = (cord[0] + 1, cord[1], cord[2] - 1)
        
        upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
        upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
        le = ind(left[0], left[1], left[2], maxSide)
        ri = ind(right[0], right[1], right[2], maxSide)
        downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
        downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
        
        neigh = [upL, upR, le, ri, downL, downR]
        emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
        
        # Iterate over every open position for every color
        for color in range(3):
            for item in emptyN:
                # If the move is a loser, decrease points
                if loser(item, color + 1, board):
                    score = score - 10
                else:
                    #save move otherwise
                    move = [color+1, item[0], item[1]]
    else:
        # If no neighbors empty or no last play, iterate over every posiiton
        for color in range(3):
            for arr in range(1, len(board)-1):
                for space in range(1, len(board[arr]) -1):
                    # If the move is a loser, decrease points
                    if loser([arr, space], color + 1, board):
                            score = score - 10
                    else:
                         #save move otherwise
                        move = [color+1, arr, space]
    # return score and valid move
    return score, move


def play(a):
    """
    input: string | ex: '[13][302][1003][30002][100003][3000002][121212]LastPlay:null'
    output: string | ex: (1,1,1,5)
    
    Formats the input from sys to fit our functions
    Then formats the output for stdout
    """
    
    # Depth of 6
    look = 6
    # Turn the input to a board
    b = stringToBoard(a)
    # Cut the last play from the input
    c = stringToPlay(a)
    # score array to pass in to tryAgain
    s =  [[[0 for val in range(1, len(b[i]) -1)] for i in range(1, len(b) - 1)] for j in range(3)]
    if c is None:
        # Find a valid move
        temp = tryAgain(b, None, s, look, float("-inf"), float("inf"), True)
        # find the coordinates and put it into output format
        cord = coord(temp[1][1:], b)
        tup = (temp[1][0], cord[2], cord[0], cord[1])
        return str(tup)

    else:
        # Change last play form
        d = playToLast(c)
        # Find a valid move
        temp = tryAgain(b, d, s, look, float("-inf"), float("inf"), True)
        # find the coordinates and put it into output format
        cord = coord(temp[1][1:], b)
        tup =  (temp[1][0], cord[2], cord[0], cord[1])
        return str(tup)



sys.stdout.write(play(sys.argv[1]));


"""

def static(board, last, lookahead):
    
    scores = [[[0 for val in range(1, len(board[i]) -1)] for i in range(1, len(board) - 1)] for j in range(3)]
    
#    print(scores)
    if (last is None) or (isSpace(last, board) == False):
        for color in range(3):
            for arr in range(1, len(board)-1):
                for space in range(1, len(board[arr]) -1):
    #               print(arr, space)
    #               print(board[arr][space])
                    if board[arr][space] != 0:
                        #print("no go")
                        scores[color][arr-1][space-1] = -10
                    elif loser([arr, space], color + 1, board):
                        #print(board)
                        #print("loser")
                        scores[color][arr-1][space-1] = 0
                    elif lookahead == 0:
                        #print("look")
                        scores[color][arr-1][space-1] = 50
                    else:
                        #print("recurssion")
                        board[arr][space] = color + 1
                        move = [arr, space]
                        opp = static(board, move, lookahead-1)
                        #print(np.amax(opp))
                        maxi = np.amax(np.amax(opp))
                        if maxi == 0:
                            #print("rec lose")
                            scores[color][arr-1][space-1] = 100
                        elif maxi == 50:
                            #print("neutral")
                            scores[color][arr-1][space-1] = 50
                        board[arr][space] = 0
    else:
        scores = neighborSearch(last, board, scores, lookahead)
                        
    return scores



def neighborSearch(last, board, scores, lookahead):
    
    cord = coord(last, board)
    maxSide = len(board) - 2
                 
    upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
    upRight = (cord[0], cord[1] - 1, cord[2] + 1)
    left = (cord[0] - 1, cord[1] + 1, cord[2])
    right = (cord[0] + 1, cord[1] - 1, cord[2])
    downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
    downRight = (cord[0] + 1, cord[1], cord[2] - 1)
    
    upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
    upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
    le = ind(left[0], left[1], left[2], maxSide)
    ri = ind(right[0], right[1], right[2], maxSide)
    downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
    downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
    
    neigh = [upL, upR, le, ri, downL, downR]
    emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
    
    for color in range(3):
        for arr in range(1, len(board)-1):
            for space in range(1, len(board[arr]) -1):
                if (board[arr][space] != 0) or ([arr, space] not in emptyN):
                    scores[color][arr-1][space-1] = -10
                elif loser([arr, space], color + 1, board):
                    scores[color][arr-1][space-1] = 0
                elif lookahead == 0:
                    scores[color][arr-1][space-1] = 50
                else:
                    board[arr][space] = color + 1
                    move = [arr, space]
                    opp = static(board, move, lookahead-1)
                    maxi = np.amax(np.amax(opp))
                    if maxi == 0:
                        scores[color][arr-1][space-1] = 100
                    elif maxi == 50:
                        scores[color][arr-1][space-1] = 50
                    board[arr][space] = 0
    return scores



def pruning(board, last, lookahead, alpha, beta, maximizingPlayer):
    
    scores = [[[0 for val in range(1, len(board[i]) -1)] for i in range(1, len(board) - 1)] for j in range(3)]
    
#    print(scores)

    if (last is None) or (isSpace(last, board) == False):
        for color in range(3):
            for arr in range(1, len(board)-1):
                for space in range(1, len(board[arr]) -1):
    #               print(arr, space)
    #               print(board[arr][space])
                    if board[arr][space] != 0:
                        #print("no go")
                        scores[color][arr-1][space-1] = -10
                    elif loser([arr, space], color + 1, board):
                        #print(board)
                        #print("loser")
                        scores[color][arr-1][space-1] = 0
                    elif lookahead == 0:
                        #print("look")
                        scores[color][arr-1][space-1] = 50
                    else:
                        #print("recurssion")
                        board[arr][space] = color + 1
                        move = [arr, space]
                        #opp = static(board, move, 0)
                        ex = False
                        if maximizingPlayer:
                            v = float("-inf")
                                                        
                            
                            #WTF DO THESE FORLOOPS DO????????????????????????
                            
                            
                            
                            for c in range(3):
                                for a in range(len(scores[c])):
                                    for val in range(len(scores[c][a])):
                                        v = max(v, pruning(board, [a+1, val+1], lookahead - 1, alpha, beta, False)[c][a][val])
                                        fi.write("max: " + str(v) + "\n")
                                        alpha = max(alpha, v)
                                        if beta <= alpha:
                                            ex = True
                                            break
                                    if ex:  
                                        break
                                if ex:
                                    ex = False
                                    break
                            scores[color][arr-1][space-1] = v
                        else:
                            v = float("inf")
                            for c in range(3):
                                for a in range(len(scores[c])):
                                    for val in range(len(scores[c][a])):
                                        v = min(v, pruning(board, [a+1, val+1], lookahead - 1, alpha, beta, True)[c][a][val])
                                        fi.write("min: " + str(v) + "\n")
                                        beta = min(beta, v)
                                        if beta <= alpha:
                                            ex = True
                                            break
                                    if ex:  
                                        break
                                if ex:  
                                    ex = False
                                    break
                            scores[color][arr-1][space-1] = v
                        board[arr][space] = 0
#                            #print(np.amax(opp))
#                            maxi = np.amax(np.amax(opp))
#                            if maxi == 0:
#                                #print("rec lose")
#                                scores[color][arr-1][space-1] = 100
#                            elif maxi == 50:
#                                #print("neutral")
#                                scores[color][arr-1][space-1] = 50
#                            board[arr][space] = 0
    else:
        scores = neighborPruning(last, board, scores, lookahead, alpha, beta, maximizingPlayer)
                        
    return scores



def neighborPruning(last, board, scores, lookahead, alpha, beta, maximizingPlayer):
    
    cord = coord(last, board)
    maxSide = len(board) - 2
                 
    upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
    upRight = (cord[0], cord[1] - 1, cord[2] + 1)
    left = (cord[0] - 1, cord[1] + 1, cord[2])
    right = (cord[0] + 1, cord[1] - 1, cord[2])
    downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
    downRight = (cord[0] + 1, cord[1], cord[2] - 1)
    
    upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
    upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
    le = ind(left[0], left[1], left[2], maxSide)
    ri = ind(right[0], right[1], right[2], maxSide)
    downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
    downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
    
    neigh = [upL, upR, le, ri, downL, downR]
    emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
    
    for color in range(3):
        for arr in range(1, len(board)-1):
            for space in range(1, len(board[arr]) -1):
                if (board[arr][space] != 0) or ([arr, space] not in emptyN):
                    scores[color][arr-1][space-1] = -10
                elif loser([arr, space], color + 1, board):
                    scores[color][arr-1][space-1] = 0
                elif lookahead == 0:
                    scores[color][arr-1][space-1] = 50
                else:
                    board[arr][space] = color + 1
                    move = [arr, space]
                    #opp = static(board, move, 0)
                    ex = False
                    if maximizingPlayer:
                        v = float("-inf")
                        for c in range(3):
                            for a in range(len(scores[c])):
                                for val in range(len(scores[c][a])):
                                    v = max(v, pruning(board, [a+1, val+1], lookahead - 1, alpha, beta, False)[c][a][val])
                                    fi.write("maxN: " + str(v) + "\n")
                                    alpha = max(alpha, v)
                                    if beta <= alpha:
                                        ex = True
                                        break
                                if ex:  
                                    break
                            if ex:
                                ex = False
                                break
                        scores[color][arr-1][space-1] = v
                    else:
                        v = float("inf")
                        for c in range(3):
                            for a in range(len(scores[c])):
                                for val in range(len(scores[c][a])):
                                    v = min(v, pruning(board, [a+1, val+1], lookahead - 1, alpha, beta, True)[c][a][val])
                                    fi.write("minN: " + str(v) + "\n")
                                    beta = min(beta, v)
                                    if beta <= alpha:
                                        ex = True
                                        break
                                if ex:  
                                    break
                            if ex:  
                                ex = False
                                break
                        scores[color][arr-1][space-1] = v
                    board[arr][space] = 0
#                    maxi = np.amax(np.amax(opp))
#                    if maxi == 0:
#                        scores[color][arr-1][space-1] = 100
#                    elif maxi == 50:
#                        scores[color][arr-1][space-1] = 50
#                    board[arr][space] = 0
    return scores



def prune(board, last, lookahead):
    
    scores = [[[0 for val in range(1, len(board[i]) -1)] for i in range(1, len(board) - 1)] for j in range(3)]
    
#    print(scores)
    for color in range(3):
        for arr in range(1, len(board)-1):
            for space in range(1, len(board[arr]) -1):
                pos = [color, arr, space]
                scores[color][arr-1][space-1] = prun(board, last, pos, scores, lookahead, float("-inf"), float("inf"), True)

    return scores



def prun(board, last, pos, scores, lookahead, alpha, beta, maximizingPlayer):
    
    #board = copy.deepcopy(boa)
    color = pos[0]
    arr = pos[1]
    space = pos[2]
    
    if (last is not None) and (isSpace(last, board)):
        
        cord = coord(last, board)
        maxSide = len(board) - 2
                     
        upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
        upRight = (cord[0], cord[1] - 1, cord[2] + 1)
        left = (cord[0] - 1, cord[1] + 1, cord[2])
        right = (cord[0] + 1, cord[1] - 1, cord[2])
        downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
        downRight = (cord[0] + 1, cord[1], cord[2] - 1)
        
        upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
        upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
        le = ind(left[0], left[1], left[2], maxSide)
        ri = ind(right[0], right[1], right[2], maxSide)
        downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
        downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
        
        neigh = [upL, upR, le, ri, downL, downR]
        emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
        #print([arr, space], emptyN)
        if ([arr, space] not in emptyN):
            #print(emptyN, board)
            return -10
    
    #if lookahead == 0: print([arr, space], loser([arr, space], color + 1, board))
        
    if board[arr][space] != 0:
        #print("no go")
        return -10
    elif loser([arr, space], color + 1, board):
        #print(board)
        #print("loser")
        #print(color + 1)
        return 0
    elif lookahead == 0:
        #print("look")
        #print("yay")
        #print(pos)
        neigh = numNeigh(pos, board)
        return 50 * len(neigh)
    else:
        #print("recurssion")
        board[arr][space] = color + 1
        la = [arr, space]
        #opp = static(board, move, 0)
        #print(color)
        ex = False
        if maximizingPlayer:
            v = float("-inf")
            for c in range(3):
                for a in range(len(scores[c])):
                    for val in range(len(scores[c][a])):
                        ######################################################
                        
                        
                        cord = coord(la, board)
                        maxSide = len(board) - 2
                                     
                        upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
                        upRight = (cord[0], cord[1] - 1, cord[2] + 1)
                        left = (cord[0] - 1, cord[1] + 1, cord[2])
                        right = (cord[0] + 1, cord[1] - 1, cord[2])
                        downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
                        downRight = (cord[0] + 1, cord[1], cord[2] - 1)
                        
                        upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
                        upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
                        le = ind(left[0], left[1], left[2], maxSide)
                        ri = ind(right[0], right[1], right[2], maxSide)
                        downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
                        downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
                        
                        neigh = [upL, upR, le, ri, downL, downR]
                        emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
                        
                        if ([a+1, val+1] not in emptyN) or board[a+1][val+1] != 0:
                            #print([a+1, val+1])
                            continue
                        
                        
                        ######################################################
                        #print("anything")
                        temp = prun(board, la, [c, a+1, val+1], scores, lookahead - 1, alpha, beta, True)
                        #if lookahead == 1: print("max: " + str(temp))
                        v = max(v, temp)
                        #if v == 0: fi.write("max: " + str(lookahead) + "\n")
                        #fi.write("max: " + str(v) + "\n")
                        alpha = max(alpha, v)
#                        if beta <= alpha:
#                            #print("max: " + str(alpha) + ", " + str(beta) )
#                            ex = True
#                            break
                    if ex:  
                        break
                if ex:
                    ex = False
                    break
            board[arr][space] = 0
            #if lookahead == 1 and pos == [1, 3, 2]: print("maxV: " + str(v))
            #if lookahead == 3 and pos == [1, 3, 2]: print("maxV3: " + str(v))
            return v
        else:
            v = float("inf")
            for c in range(3):
                for a in range(len(scores[c])):
                    for val in range(len(scores[c][a])):
                        ######################################################
                        
                        
                        cord = coord(la, board)
                        maxSide = len(board) - 2
                                     
                        upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
                        upRight = (cord[0], cord[1] - 1, cord[2] + 1)
                        left = (cord[0] - 1, cord[1] + 1, cord[2])
                        right = (cord[0] + 1, cord[1] - 1, cord[2])
                        downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
                        downRight = (cord[0] + 1, cord[1], cord[2] - 1)
                        
                        upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
                        upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
                        le = ind(left[0], left[1], left[2], maxSide)
                        ri = ind(right[0], right[1], right[2], maxSide)
                        downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
                        downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
                        
                        neigh = [upL, upR, le, ri, downL, downR]
                        emptyN = [i for i in neigh if board[i[0]][i[1]] == 0]
                        
                        if ([a+1, val+1] not in emptyN) or board[a+1][val+1] != 0:
                            continue
                        
                        
                        ######################################################
                        #print("any")
                        temp = prun(board, la, [c, a+1, val+1], scores, lookahead - 1, alpha, beta, True)
                        #if lookahead == 1: print("min: " + str(temp))
                        v = min(v, temp)
                        #print(v)
                        #if v == 0: fi.write("min: " + str(lookahead) + "\n")
                        #fi.write("min: " + str(v) + "\n")
                        beta = min(beta, v)
#                        if beta <= alpha:
#                            #print("min: " + str(alpha) + ", " + str(beta) )
#                            ex = True
#                            break
                    if ex:  
                        break
                if ex:  
                    ex = False
                    break
            board[arr][space] = 0
            #if lookahead == 2: print("minV: " + str(v))
            return v
            
            

def numNeigh(position, board):
    
    pos = position[1:]
    cord = coord(pos, board)
    maxSide = len(board) - 2
                 
    upLeft = (cord[0] - 1, cord[1], cord[2] + 1)
    upRight = (cord[0], cord[1] - 1, cord[2] + 1)
    left = (cord[0] - 1, cord[1] + 1, cord[2])
    right = (cord[0] + 1, cord[1] - 1, cord[2])
    downLeft = (cord[0], cord[1] + 1, cord[2] - 1)
    downRight = (cord[0] + 1, cord[1], cord[2] - 1)
    
    upL = ind(upLeft[0], upLeft[1], upLeft[2], maxSide)
    upR = ind(upRight[0], upRight[1], upRight[2], maxSide)
    le = ind(left[0], left[1], left[2], maxSide)
    ri = ind(right[0], right[1], right[2], maxSide)
    downL = ind(downLeft[0], downLeft[1], downLeft[2], maxSide)
    downR = ind(downRight[0], downRight[1], downRight[2], maxSide)
    
    neigh = [upL, upR, le, ri, downL, downR]
    if len(neigh) > 0:
        #print(neigh)
        return [i for i in neigh if board[i[0]][i[1]] != 0]
    else:
        return [0]

"""
