# Tensorflow
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

"""
Return int array of board numbers
"""

def gen_board(fen_s):

    list_ls = fen_s.split("/")
    board_li = [0 for i in range(65)] 

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

                v_c = rank_s[spot_i]
                coef_i = -1 if v_c.islower() else 1
                if v_c == "Q" or v_c == "q":
                    board_li[i * 8 + column_i] = .9 * coef_i
                elif v_c == "R" or v_c == "r":
                    board_li[i * 8 + column_i] = .5 * coef_i
                elif v_c == "B" or v_c == "b":
                    board_li[i * 8 + column_i] = .3 * coef_i
                elif v_c == "N" or v_c == "n":
                    board_li[i * 8 + column_i] = .3 * coef_i
                elif v_c == "P" or v_c == "p":
                    board_li[i * 8 + column_i] = .1 * coef_i
                else:
                    board_li[i * 8 + column_i] = 1 * coef_i
                column_i += 1; spot_i += 1
            
    return board_li

"""
"""

def make_new(file_file, inputs_2di, outputs_2di):

    model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu', input_shape=(65)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(inputs_2di, outputs_2di, epochs=100) 

    model.save(file_file)

"""
"""

def train_ex(file_file, inputs_2di, outputs_2di):

    model = keras.models.load_model(file_file)
    model.fit(inputs_2di, outputs_2di, epochs=100) 
    model.save(file_file)

"""
"""

def main():

    ninputs_i = 65; nex_i = 1000000
    inputs_2di = [[0 for j in range(65)] for i in range(nex_i)]
    outputs_2di = [[0 for j in range(1)] for i in range(nex_i)]

    with open("chess_train.csv") as data_file:
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

            ex_i += 1; 
            if ex_i == nex_i: break

        #train_ex("cengine_model", inputs_2di, outputs_2di) # 1,000,000 examples
        #train_ex("lowlevel_cengine_model", inputs_2di, outputs_2di) # 100,000 examples
        make_new("cengine_model",  inputs_2di, outputs_2di)

    return

main()
        


            




