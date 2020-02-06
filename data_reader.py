import scipy.io as sio
import numpy as np
import pandas as pd

NUM_WELLS = 20
NUM_MODELS = 104

def load_data(data_dir):
    '''
    load .mat file and return the content
    # INPUT: .mat file
    # OUTPUT: content of input .mat file
    '''
    mat_content = sio.loadmat(data_dir)
    return mat_content['en_d'][0, 0]

def read_data(data):
    '''
    read dataset and convert it to dictionary
    # INPUT: content of .mat file
    # OUTPUT: dictionary contains Dataframe of each well/model
    { # well_number
        '1'   => { # model_number
            '1'     => Dataframe,
                ...
            '104'   => Dataframe 
        },
        '2'   => {
            ...
        },
        ...
        '104' => {
            ...
        }
    }
    '''
    well_dic = {}
    for well_index in range(NUM_WELLS):
        model_dic = {}
        well_key = 'P' + str(well_index + 1)

        for model_index in range(NUM_MODELS):    
            well_data = np.array([
                data['WOPR'][0, 0][well_key][:,model_index],
                data['WBHP'][0, 0][well_key][:,model_index],
                data['WWCT'][0, 0][well_key][:,model_index],
                data['WWPR'][0, 0][well_key][:,model_index]
                ])
            # col1: WOPR, col2: WBHP, col3: WWCT, col4: WWPR
            # row1: day1, ... row 498: day3648
            well_data = well_data.T
            df = pd.DataFrame(
                data=well_data,
                index=data['TIME'].flatten(),
                columns=['WOPR', 'WBHP', 'WWCT', 'WWPR']
                )
            df.index.name = 'date'
            model_dic[str(model_index+1)] = df
        
        well_dic[str(well_index+1)] = model_dic

    return well_dic

def choose_well_and_model(data_dic, well_num, model_num):
    '''
    returns data of given well and model
    INPUT: dataset dictionary, well number as string, model_number as string
    OUTPUT: data in numpy.ndarray
    '''
    return data_dic[well_num][model_num].values