import chess
import cengine
import random

def predict_pos(fen_s):

    board_li = [[0 for j in range(66)] for i in range(1)]
    m_c = fen_s[fen_s.index(" ") + 1 : fen_s.index(" ") + 2]
    fen_s = fen_s[0 : fen_s.index(" ")]
    board_li[0] = cengine.gen_board(fen_s)
    board_li[0][64] = -1 if m_c == "b" else 1

    return board_li[0][65] + random.randint(-2, 2) / 4

def beam_search(state, depth, ismax):

    backtrack = {}
    eval = {}
    pred = None
    
    for nm in state.legal_moves:

        state.push(nm)
        new_state = state.copy()
        res = 0; count = 0

        for lm in new_state.legal_moves:

            new_state.push(lm)
            last_state = new_state.copy()
            res += (predict_pos(last_state.fen())) ** 2
            count += 1
            new_state.pop()

        eval[str(nm) + "!"] = [res / count, new_state]
        state.pop()
    
    r_moves = sorted(eval.keys(), key = lambda x: eval[x][0], reverse = True if ismax else False)[ : 4]
    r_values =  sorted(eval.values(), key = lambda x: x[0])[ : 4]
    q = []
    for i in range(4): 
        backtrack[r_moves[i]] = pred
        q.append([r_moves[i], r_values[i]])

    return q


def main():

    board = chess.Board()
    global posmem; global model
    posmem = {}
    topevals_lf = []
    #model = keras.models.load_model("model_data/" + sys.argv[1])
    white_b = True

    print(beam_search(board, 0, white_b))

    """
    while (True):

        topevals_lf.clear()
        posmem.clear()
        t = time.perf_counter()
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
        """

if __name__ == "__main__":
    main()
