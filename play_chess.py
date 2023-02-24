import sys

import chess
import cengine

from tensorflow import keras

def predict_pos(fen_s):

    board_li = [[0 for j in range(66)] for i in range(1)]
    m_c = fen_s[fen_s.index(" ") + 1 : fen_s.index(" ") + 2]
    fen_s = fen_s[0 : fen_s.index(" ")]
    board_li[0] = cengine.gen_board(fen_s)
    board_li[0][64] = -1 if m_c == "b" else 1

    #return model.predict(board_li, verbose = 0)
    return board_li[0][65]

def possible_new_states(state):

    ret = []
    count = 0
    for i in state.legal_moves:
        state.push(i)
        ret.append(state.copy())
        state.pop()

    return ret

def minimax_ab(state, depth, is_maximizing, alpha, beta):

    if depth == 0:
        return predict_pos(state.fen())

    if is_maximizing:
        bV = -30
        for new_state in possible_new_states(state):
            if new_state.fen() in pos_eval:
                value = pos_eval[new_state.fen()]
            else:
                value = minimax_ab(new_state, depth - 1, False, alpha, beta)
                pos_eval[new_state.fen()] = value
            bV = max(bV, value) 
            alpha = max(alpha, bV)
            if beta <= alpha:
                break
        return bV

    else:
        bV = 30
        for new_state in possible_new_states(state):
            if new_state.fen() in pos_eval:
                value = pos_eval[new_state.fen()]
            else:
                value = minimax_ab(new_state, depth - 1, True, alpha, beta)
                pos_eval[new_state.fen()] = value
            bV = min(bV, value) 
            beta = min(beta, bV)
            if beta <= alpha:
                break
        return bV

def minimax(state, depth, is_maximizing):

    if depth == 0:
        return predict_pos(state.fen())

    if is_maximizing:
        bV = -30
        for new_state in possible_new_states(state):
            if new_state.fen() in pos_eval:
                value = pos_eval[new_state.fen()]
            else:
                value = minimax(new_state, depth - 1, False)
                pos_eval[new_state.fen()] = value
            bV = max(bV, value) 
        return bV

    else:
        bV = 30
        for new_state in possible_new_states(state):
            if new_state.fen() in pos_eval:
                value = pos_eval[new_state.fen()]
            else:
                value = minimax(new_state, depth - 1, True)
                pos_eval[new_state.fen()] = value
            bV = min(bV, value) 
        return bV

def main():

    board = chess.Board()
    global model
    global pos_eval
    pos_eval = {}
    topevals_lf = []
    model = keras.models.load_model("model_data/" + sys.argv[1])
    topevals_lf = [[0 for j in range (2)] for i in range(4)]
    white_b = True

    while (True):

        print(board)
        print("Top Moves: ", end = '')
        topevals_lf.clear()
        pos_eval.clear()

        for i in board.legal_moves:

            board.push(i)
            #eval_f = minimax(board, 3, white_b)
            eval_f = minimax_ab(board, 1, white_b, -30, 30)
            topevals_lf.append([eval_f, i])
            board.pop()
        
        print(len(topevals_lf), " | ", len(pos_eval))
        topevals_lf.sort(reverse=True, key = lambda x: x[0]) if white_b else topevals_lf.sort(key = lambda x: x[0])
        for i in range(min(4, len(topevals_lf))):
            print(i+1, end = ""); print(". ", end = ""); print(topevals_lf[i][1], topevals_lf[i][0], " ", end = "")

        move = input("\nEnter Move: ")
        if move == "quit": 
            break
        board.push(chess.Move.from_uci(move))
        white_b = False if white_b else True


if __name__ == "__main__":
    main()
