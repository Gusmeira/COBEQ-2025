import pandas as pd
import numpy as np




ponte = pd.read_excel(r'Data\marg.tietê-ponte-dos remédios, são paulo, brazil-air-quality.xlsx')
ponte['date'] = pd.to_datetime(ponte['date'], format="%d/%m/%Y")
ponte = ponte.fillna(np.nan)

ponte = ponte[(ponte['date'] >= '2016-04-01') & (ponte['date'] <= '2022-12-31')]
ponte = ponte.sort_values(by='date')

ponte['pm25'] = pd.to_numeric(ponte['pm25'], errors='coerce')
ponte['pm10'] = pd.to_numeric(ponte['pm10'], errors='coerce')
ponte['o3'] = pd.to_numeric(ponte['o3'], errors='coerce')
ponte['no2'] = pd.to_numeric(ponte['no2'], errors='coerce')
ponte['so2'] = pd.to_numeric(ponte['so2'], errors='coerce')
ponte['co'] = pd.to_numeric(ponte['co'], errors='coerce')

ponte.to_pickle(r'Data\Data_Ponte_dos_Remedios.pkl')




guarulhos = pd.read_excel(r'Data\guarulhos-paço-municipal, são paulo, brazil-air-quality.xlsx')
guarulhos['date'] = pd.to_datetime(guarulhos['date'], format="%d/%m/%Y")
guarulhos = guarulhos.fillna(np.nan)

guarulhos = guarulhos[(guarulhos['date'] >= '2016-04-01') & (guarulhos['date'] <= '2022-12-31')]
guarulhos = guarulhos.sort_values(by='date')

guarulhos['pm25'] = pd.to_numeric(guarulhos['pm25'], errors='coerce')
guarulhos['pm10'] = pd.to_numeric(guarulhos['pm10'], errors='coerce')
guarulhos['o3'] = pd.to_numeric(guarulhos['o3'], errors='coerce')
guarulhos['no2'] = pd.to_numeric(guarulhos['no2'], errors='coerce')
guarulhos['so2'] = pd.to_numeric(guarulhos['so2'], errors='coerce')
guarulhos['co'] = pd.to_numeric(guarulhos['co'], errors='coerce')

guarulhos.to_pickle(r'Data\Data_Guarulhos.pkl')