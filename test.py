import sys
import time
import chess
import cengine
from tensorflow import keras
from multiprocessing import Pool

class datap():

    def __init__(self, s, d, i, a, b, m):
        self.state = s; self.depth = d; self.ismax = i; self.alpha = a; self.beta = b; self.model = m

    state: None
    depth: int
    ismax: bool
    alpha: int
    beta: int
    model: None

def possible_new_states(state):

    ret = []
    for i in state.legal_moves:
        state.push(i)
        ret.append(state.copy())
        state.pop()

    return ret

def beam_minimax(pack):

    def predict_pos(fen_s):

        board_li = [[0 for j in range(66)] for i in range(1)]
        m_c = fen_s[fen_s.index(" ") + 1 : fen_s.index(" ") + 2]
        fen_s = fen_s[0 : fen_s.index(" ")]
        board_li[0] = cengine.gen_board(fen_s)
        board_li[0][64] = -1 if m_c == "b" else 1

        return pack.model.predict(board_li, verbose = 0) + board_li[0][65] / 2

    alpha = pack.alpha; beta = pack.beta; is_maximizing = pack.ismax; depth = pack.depth; state = pack.state

    if depth == 0:
        return predict_pos(pack.state.fen())
    
    if is_maximizing:

        bV = -30
        q = []
        for new_state in possible_new_states(state):
            q.append([predict_pos(new_state.fen()), new_state])
        q.sort(reverse=True, key = lambda x: x[0])
        for i in range(min(depth, len(q))):
            value = beam_minimax(datap(q[i][1], depth - 1, False, alpha, beta, pack.model))
            bV = max(bV, value) 
            alpha = max(alpha, bV)
            if beta <= alpha:
                break
        return bV
    
    else:
        
        bV = 30
        q = []
        for new_state in possible_new_states(state):
            q.append([predict_pos(new_state.fen()), new_state])
        q.sort(key = lambda x: x[0])
        for i in range(min(depth, len(q))):
            value = beam_minimax(datap(q[i][1], depth - 1, True, alpha, beta, pack.model))
            bV = min(bV, value) 
            beta = min(beta, bV)
            if beta <= alpha:
                break
        return bV

def main():

    board = chess.Board()
    topevals_lf = []
    model = keras.models.load_model("model_data/" + sys.argv[1])
    white_b = True

    while (True):

        topevals_lf.clear()

        t = time.perf_counter()

        q = []; mv = []
        for i in board.legal_moves:
            board.push(i)
            mv.append(i)
            q.append(datap(board.copy(), 4, not white_b, -30, 30, model))
            board.pop()

        with Pool() as pool:
            res = pool.map(beam_minimax, q)
            for p, v in enumerate(res):
                topevals_lf.append([v, mv[p]])

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
