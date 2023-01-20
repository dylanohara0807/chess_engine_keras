import sys

import chess
import cengine

import tensorflow as tf
from tensorflow import keras

"""
"""

def predict_pos(fen_s, model):

    board_li = [[0 for j in range(65)] for i in range(1)]
    m_c = fen_s[fen_s.index(" ") + 1 : fen_s.index(" ") + 2]
    fen_s = fen_s[0 : fen_s.index(" ")]
    board_li[0] = cengine.gen_board(fen_s)
    board_li[0][64] = -1 if m_c == "b" else 1

    return model.predict(board_li, verbose = 0)


"""
"""

def main():

    board = chess.Board()
    model = keras.models.load_model(sys.argv[1])
    topevals_lf = [[0 for j in range (2)] for i in range(4)]
    white_b = 1

    while (True):

        print(board)
        print("Top Moves: ", end = '')

        count_i = 0
        for i in board.legal_moves:

            board.push(i)
            eval_f = predict_pos(board.fen(), model)

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
