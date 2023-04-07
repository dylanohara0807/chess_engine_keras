import sys
import time
import chess
import cengine
import numpy as np
from tensorflow import keras

def predict_pos(fen_s):

    board_li = [[0 for j in range(66)] for i in range(1)]
    m_c = fen_s[fen_s.index(" ") + 1 : fen_s.index(" ") + 2]
    fen_s = fen_s[0 : fen_s.index(" ")]
    board_li[0] = cengine.gen_board(fen_s)
    board_li[0][64] = -1 if m_c == "b" else 1

    return model.predict(board_li, verbose = 0) #+ board_li[0][65] / 3

def possible_new_states(state):

    ret = []
    for i in state.legal_moves:
        state.push(i)
        ret.append(state.copy())
        state.pop()

    return ret

def beam_minimax(state, depth, is_maximizing, alpha, beta):

    if depth == 0:
        return predict_pos(state.fen())

    if is_maximizing:
        bV = -30
        q = []
        for new_state in possible_new_states(state):
            if new_state.fen() in posmem:
                q.append([posmem[new_state.fen()], new_state])
            else:
                q.append([predict_pos(new_state.fen()), new_state])

        q.sort(reverse=True, key = lambda x: x[0])
        for i in range(min(depth, len(q))):
            if q[i][1].fen() in posmem:
                value = q[i][0]
            else:
                value = beam_minimax(q[i][1], depth - 1, False, alpha, beta)
                posmem[q[i][1].fen()] = value
            bV = max(bV, value) 
            alpha = max(alpha, bV)
            if beta <= alpha:
                break
        return bV

    else:
        bV = 30
        q = []
        for new_state in possible_new_states(state):
            if new_state.fen() in posmem:
                q.append([posmem[new_state.fen()], new_state])
            else:
                q.append([predict_pos(new_state.fen()), new_state])
        q.sort(key = lambda x: x[0])
        for i in range(min(depth, len(q))):
            if q[i][1].fen() in posmem:
                value = q[i][0]
            else:
                value = beam_minimax(q[i][1], depth - 1, True, alpha, beta)
                posmem[q[i][1].fen()] = value
            bV = min(bV, value) 
            beta = min(beta, bV)
            if beta <= alpha:
                break
        return bV

def main():

    board = chess.Board()
    global posmem; global model
    posmem = {}
    topevals_lf = []
    model = keras.models.load_model("model_data/" + sys.argv[1])
    white_b = True

    while (True):

        topevals_lf.clear()
        posmem.clear()

        t = time.perf_counter()
        for i in board.legal_moves:
            board.push(i)
            eval_f = beam_minimax(board, 4, not white_b, -30, 30)
            topevals_lf.append([eval_f, i])
            board.pop()
        print("Time Alloted:", time.perf_counter() - t)

        topevals_lf.sort(reverse=True, key = lambda x: x[0]) if white_b else topevals_lf.sort(key = lambda x: x[0])
        print(board)
        print("Top Moves: ", end = '')
        for i in range(min(4, len(topevals_lf))):
            print(i+1, end = ""); print(". ", end = ""); print(topevals_lf[i][1], topevals_lf[i][0], end = " | ")
        while True:
            try:
                move = input("\nEnter Move: ")
                if move == "quit": exit()
                board.push(chess.Move.from_uci(move))
                break
            except Exception:
                print("Invalid move.",end="")
        white_b = False if white_b else True

if __name__ == "__main__":
    main()
