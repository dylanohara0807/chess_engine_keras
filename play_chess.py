import sys

import chess
import cengine

import tensorflow as tf
from tensorflow import keras

def predict_pos(fen_s):

    board_li = [[0 for j in range(66)] for i in range(1)]
    m_c = fen_s[fen_s.index(" ") + 1 : fen_s.index(" ") + 2]
    fen_s = fen_s[0 : fen_s.index(" ")]
    board_li[0] = cengine.gen_board(fen_s)
    board_li[0][64] = -1 if m_c == "b" else 1

    return model.predict(board_li, verbose = 0)

def possible_new_states(state):

    ret = []
    for i in state.legal_moves:
        state.push(i)
        ret.append(state)
        state.pop()

    return ret

def evaluate(state):

    return predict_pos(state.fen())

def minimax(state, depth, is_maximizing, alpha, beta):

    if depth == 0:
        return evaluate(state)

    if is_maximizing:
        bV = -30
        for new_state in possible_new_states(state):
            value = minimax(new_state, depth - 1, False, alpha, beta)
            bV = max(bV, value) 
            alpha = max(alpha, bV)
            if beta <= alpha:
                break
        return bV

    else:
        bV = 30
        for new_state in possible_new_states(state):
            value = minimax(new_state, depth - 1, True, alpha, beta)
            bV = min(bV, value) 
            beta = min(beta, bV)
            if beta <= alpha:
                break
        return bV

def main():

    board = chess.Board()
    global model
    model = keras.models.load_model("model_data/" + sys.argv[1])
    topevals_lf = [[0 for j in range (2)] for i in range(4)]
    white_b = 1

    while (True):

        print(board)
        print("Top Moves: ", end = '')
        
        count_i = 0
        for i in board.legal_moves:

            board.push(i)
            eval_f = minimax(board, 3, True if white_b == 1 else False, -30, 30)

            if count_i < 4:

                topevals_lf[count_i][0] = eval_f
                topevals_lf[count_i][1] = i
            
            else:

                if (white_b):

                    pos_i = 3
                    for j in range(3):
                        if topevals_lf[j][0] < topevals_lf[pos_i][0]:
                            pos_i = j
                    if eval_f > topevals_lf[pos_i][0]:
                        topevals_lf[pos_i][0] = eval_f
                        topevals_lf[pos_i][1] = i
                
                else:

                    pos_i = 3
                    for j in range(3):
                        if topevals_lf[j][0] > topevals_lf[pos_i][0]:
                            pos_i = j
                    if eval_f < topevals_lf[pos_i][0]:
                        topevals_lf[pos_i][0] = eval_f
                        topevals_lf[pos_i][1] = i

            count_i += 1
            board.pop()

        for i in range(4):
            print(i+1, end = ""); print(". ", end = ""); print(topevals_lf[i][1], topevals_lf[i][0], " ", end = "")
        
        move = input("\nEnter Move: ")
        if move == "quit": 
            break
    
        board.push(chess.Move.from_uci(move))
        white_b = 0 if white_b == 1 else 1


if __name__ == "__main__":
    main()
