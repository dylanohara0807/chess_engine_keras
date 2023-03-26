from tensorflow import keras
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np

import sys

global piece_value
piece_value = {"Q": 9, "q": -9, "R": 5, "r": -5, "B": 3, "b": -3, 
                "N": 3, "n": -3, "P": 1, "p": -1, "K": 10, "k": -10}
def gen_board(fen_s):
    """ 
    Generate a chess board in the form of a float array, given a fen
    representation of the board.

    Args:
        fen_s (str): fen format chess state (rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR).

    Returns:
        float[66]: array of each chess piece value(/10) at each given position (0 if no piece, 1 if king).
        Two extra extra spots, one for side to move, and one for total material balance (not /10).
    """        

    list_ls = fen_s.split("/")
    board_li = [0 for i in range(66)] 
    material_i = 0
    
    for i in range(8):

        column_i = 0; spot_i = 0
        rank_s = list_ls[i]

        while(column_i < 8):

            spaces_i = 0
            if rank_s[spot_i].isdigit():

                spaces_i = (int) (rank_s[spot_i])
                for j in range(column_i, column_i + spaces_i): 
                    board_li[i * 8 + j] = 0
                column_i += spaces_i; spot_i += 1

            else:

                p_v = piece_value[rank_s[spot_i]]
                board_li[i * 8 + column_i] = p_v / 10
                material_i += p_v
                column_i += 1; spot_i += 1
    
    board_li[65] = material_i
    return board_li

def make_new(file_file, inputs_2di, outputs_2di):
    """ 
    Create a train a new chess_engine, save it to directory file_file.

    Args:
        file_file (str): File name to save traind model to.
        inputs_2di (float[][66]): 2D array of the inputs.
        outputs_2di (float[][1]): 2D array of outputs (one output value).
    """
    def stretched_tanh(x):
        return (K.tanh(x / 5) * 5)

    model = keras.Sequential([
    keras.layers.Dense(4096, activation=stretched_tanh, input_shape=[66]),
    keras.layers.Dense(1024, activation=stretched_tanh),
    keras.layers.Dense(256, activation=stretched_tanh),
    keras.layers.Dense(256, activation=stretched_tanh),
    keras.layers.Dense(256, activation=stretched_tanh),
    keras.layers.Dense(256, activation=stretched_tanh),
    keras.layers.Dense(256, activation=stretched_tanh),
    keras.layers.Dense(256, activation=stretched_tanh),
    ###
    keras.layers.Dense(64, activation=stretched_tanh),
    keras.layers.Dense(64, activation=stretched_tanh),
    keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer="Adam", loss='mean_squared_error')
    model.fit(inputs_2di, outputs_2di, epochs=10, batch_size=1000) 
    model.save("model_data/" + file_file)

def train_ex(file_file, inputs_2di, outputs_2di):
    """ 
    Train an already existing model, given by file_file (overrides existing model). 
    Then save the model and a plot of training/validation accurazy over time.

    Args:
        file_file (str): File name to train, and resave.
        inputs_2di (float[][66]): 2D array of the inputs.
        outputs_2di (float[][1]): 2D array of outputs (one output value).
    """     

    model = keras.models.load_model("model_data/" + file_file)
    history =  model.fit(inputs_2di, outputs_2di, epochs=10, batch_size = 1000, validation_split = .1) 

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("accuracy_graphs/" + file_file)
    plt.show()

    model.save("model_data/" + file_file)

def main():
    """ 
    Parse FEN and eval data from chess_train.csv using hardcoded length.
    Clip evaluations to [-15, 15] which is sufficient for human understanding of evaluation.
    Either train or create a new model, based on hardcoded input.

    """    

    def load_file():

        inputs_2di = [[0 for j in range(66)] for i in range(nex_i)]
        outputs_2di = [[0 for j in range(1)] for i in range(nex_i)]

        with open("training_data/chess_train.csv") as data_file:
            next(data_file)
            ex_i = 0
            for line_s in data_file:

                fen_s = line_s[0 : line_s.index(" ")]
                m_c = line_s[line_s.index(" ") + 1 : line_s.index(" ") + 2]
                if line_s[line_s.index(",") + 1] != "#":
                    eval_f = (float) (line_s[line_s.index(",") + 1 : ]) 
                else:
                    eval_f = 1500 * (float) (line_s[line_s.index(",") + 2 : ])
                eval_f /= 100
                outputs_2di[ex_i] = min(max(eval_f, -15), 15)
                inputs_2di[ex_i] = gen_board(fen_s); inputs_2di[ex_i][64] = -1 if m_c == "b" else 1

                if (False):
                    test = [[0 for j in range(65)] for i in range(1)]
                    test[0] = inputs_2di[ex_i]
                    print("Predicted: ", model.predict(test, verbose = 0))
                    print("Actual: ", outputs_2di[ex_i])

                ex_i += 1; 
                if ex_i == nex_i: break
        np.save("training_data/inputs_board", inputs_2di)
        np.save("training_data/outputs_eval", outputs_2di)

    nex_i = 12900000
    #model = keras.models.load_model(sys.argv[1])
    
    inputs_2di = np.load("training_data/inputs_board.npy")
    outputs_2di = np.load("training_data/outputs_eval.npy")

    #train_ex("adam_stretchedtanh", inputs_2di, outputs_2di)
    #make_new("adam_stretchedtanh", inputs_2di, outputs_2di)

    return

if __name__ == "__main__":
    main()
        


            




