from os.path import dirname, join
import scipy.io as sio
import numpy as np
import pandas as pd

class Reader:
    NUM_PRODUCER_WELLS = 20

    def __init__(self, data_path, data_filename, num_model):
        self._data_dir = join(data_path, data_filename)
        self._num_model = num_model

    def load_mat_content(self):
        mat_contents = sio.loadmat(self._data_dir)
        data = mat_contents['en_d'][0,0]
        return data

    def create_well_dictionary(self):
        data = self.load_mat_content()
        well_dic = {}

        for well_index in range(self.NUM_PRODUCER_WELLS): 
            # key: 'model_num' => value: dataframe
            model_dic = {}
            well_key = 'P' + str(well_index+1)

            for model_index in range(self._num_model): 
                well_data = np.array([
                    data['WOPR'][0,0][well_key][:,model_index],
                    data['WBHP'][0,0][well_key][:,model_index],
                    data['WWCT'][0,0][well_key][:,model_index],
                    data['WWPR'][0,0][well_key][:,model_index]
                  ])
                # col1: WOPR, col2: WBHP, col3: WWCT, col4: WWPR
                # row1: day1, ... row 498: day3648
                well_data = well_data.T
                df = pd.DataFrame(
                    data=well_data,
                    columns=['WOPR', 'WBHP', 'WWCT', 'WWPR']
                )
                model_dic[str(model_index+1)] = df

            well_dic[str(well_index+1)] = model_dic

        return well_dic
        