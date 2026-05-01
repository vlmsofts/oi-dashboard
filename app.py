"""
Cotton Belt Drought & Planting Dashboard  +  COT Positioning Dashboard
=======================================================================
Run:  python app.py
Then: http://127.0.0.1:8050

Tabs
----
1. Production & analogs       – cotton belt drought / production model
2. Seasonal drought profile   – D2+ trajectory vs analog years
3. Futures & planting signal  – CBOT price index + planting signal
4. State detail               – per-state production, yield penalty, acres
5. COT – About & guide        – what the COT data is and how to use this tab
6. COT – Positioning heatmap  – 14-commodity × 5-category percentile grid
7. COT – Price projections    – analog-based probability-weighted price targets

Data sources
------------
- USDA Drought Monitor dm_export_20060320_20260320.csv
- NCC 2026 Planting Intentions
- CFTC Disaggregated COT (2006-2026), 928 weeks, 14 markets
- USDA NASS county/state cotton production
"""

__version__ = "2026.03.24-fix5"
import sys
sys.dont_write_bytecode = True  # prevent stale cache issues

import json
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – ORIGINAL COTTON DATA
# ══════════════════════════════════════════════════════════════════════════════

STATES   = ['TX','GA','AR','MS','NC','AL','SC','OK','LA','TN','MO','KS','AZ','VA','FL','NM','CA']
YEARS    = list(range(2006, 2025))
NAT_PROD = [20822,18355,12395,11783,17598,14722,16534,12275,
            15753,12455,16601,20223,17566,19227,14061,17191,13998,11750,13942]
BELT_SCORE=[101,147,95,34,30,204,183,121,106,72,67,27,84,19,51,109,141,111,52]
BELT_D2  = [10.4,15.5,8.5,3.5,3.7,11.7,20.9,16.5,10.7,9.3,7.3,3.8,8.6,2.2,9.5,18.9,20.4,11.1,8.8]

FUT_YEARS= list(range(2010, 2027))
DEC_CT   = [102,110,73,84,63,63,73,78,80,66,72,108,92,82,72,68,65,65]
NOV_SOY  = [1280,1300,1500,1290,1010,990,1050,1010,890,980,1150,1490,1480,1360,1040,1000,980,980]
DEC_CORN = [570,660,540,485,390,390,370,395,380,415,430,600,630,490,465,450,460,460]

NCC_2026 = [5320,805,362,278,268,290,152,451,105,179,266,112,83,60,54,25,17]
NCC_2025 = [5300,835,520,330,285,290,170,390,90,205,355,102,87,73,61,30,18]
BASE_Y   = [668,931,1268,1080,985,877,924,675,964,1098,1295,753,1367,1124,680,853,1908]
CUR_D2   = [55.5,80.0,74.8,16.5,65.5,28.1,58.6,62.2,50.9,14.7,7.7,3.5,13.5,25.0,90.7,51.1,0.0]
CUR_D3   = [22.6,42.5,37.0,3.6,4.7,5.3,15.7,22.5,15.4,0.3,0.5,0.0,0.0,0.0,72.9,4.6,0.0]
FCAST_ADJ= [-.30,-.10,-.45,-.30,-.15,-.25,-.10,-.35,-.35,-.30,-.30,.10,.05,-.20,-.10,-.05,0.0]
FCAST_D2 = [max(0, CUR_D2[i]*(1+FCAST_ADJ[i])) for i in range(len(STATES))]
FCAST_D3 = [max(0, CUR_D3[i]*(1+FCAST_ADJ[i])) for i in range(len(STATES))]

ANALOG_YRS  = [2011,2022,2007,2012,2013]
ANALOG_D2   = [11.7,20.4,15.5,20.9,16.5]
ANALOG_PROD = [14.72,13.998,18.355,16.534,12.275]

MONTHS_LBL = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
SEAS = {
    'avg': [4,5,6,7,8,10,12,14,15,16,17,18,18],
    2026 : [12,15,18,22,28,35,40,44,46,48,49,50,49],
    2011 : [6,8,10,14,20,32,52,65,70,72,73,74,72],
    2022 : [8,10,14,18,24,30,38,45,48,50,51,52,51],
    2007 : [8,12,18,24,30,38,46,52,55,57,58,59,58],
    2012 : [5,8,12,16,22,32,44,52,56,58,59,60,59],
    2013 : [5,7,10,14,18,24,32,38,42,44,45,46,45],
}
PEN_D2 = [0,5,10,15,20,25,30,35,40,50,60,70,80,90]
PEN_YD = [0,-2,-5,-8,-10,-13,-22,-30,-34,-46,-60,-73,-81,-81]

def yield_model(d2, d3):
    if d2 < 5:  return 0,   0.965
    if d2 < 20: return -10, 0.930
    if d2 < 40: return -34, 0.900
    return          -81, 0.710

def build_state_rows(tx_fail_loss=0):
    out = []
    for i, st in enumerate(STATES):
        d2, d3 = FCAST_D2[i], FCAST_D3[i]
        pen, hr = yield_model(d2, d3)
        adj_y = max(300, BASE_Y[i] + pen + (-0.8 * d3))
        prod  = max(0, (NCC_2026[i] * hr * adj_y) / 480)
        if st == 'TX': prod = max(0, prod - tx_fail_loss)
        out.append(dict(st=st, planted=NCC_2026[i], d2=round(d2,1),
                        adj_y=round(adj_y), hr=round(hr,3), prod=round(prod)))
    return out

BASE_ROWS = build_state_rows(0)
FAIL_ROWS = build_state_rows(525)
NAT_BASE  = sum(r['prod'] for r in BASE_ROWS)
NAT_FAIL  = sum(r['prod'] for r in FAIL_ROWS)

def planting_signal():
    ct   = np.array(DEC_CT,   float)
    soy  = np.array(NOV_SOY,  float)
    corn = np.array(DEC_CORN, float)
    ct_z  = (ct  - ct.mean())  / ct.std()
    scr   = soy / corn
    scr_z = (scr - scr.mean()) / scr.std()
    return [round(float(ct_z[i] - 0.5*scr_z[i]), 2) for i in range(len(FUT_YEARS))]

SIGNALS = planting_signal()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – COT DATA  (loaded from pre-processed JSON files)
# ══════════════════════════════════════════════════════════════════════════════

# COT data — loaded from cot_data/ folder next to app.py
# Falls back to embedded Python literals so COT tabs work without data files
import os, pathlib

_HERE    = pathlib.Path(__file__).parent
_COT_DIR = pathlib.Path(os.environ.get('COT_DATA_DIR', _HERE / 'cot_data'))

# Embedded fallback data (Python literals — always available)
_EMBEDDED_HEATMAP = {'Cotton': {'OI': 327959, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': -33546.0, 'r': 9}, 'Long': {'v': 41405.0, 'r': 56}, 'Short': {'v': 74951.0, 'r': 96}, 'NetPct': {'v': -10.2, 'r': 12}, 'LongPct': {'v': 12.6, 'r': 14}, 'ShortPct': {'v': 22.9, 'r': 88}}, 'Prod': {'Net': {'v': -66930.0, 'r': 59}, 'Long': {'v': 57152.0, 'r': 91}, 'Short': {'v': 124082.0, 'r': 58}, 'NetPct': {'v': -19.6, 'r': 81}, 'LongPct': {'v': 16.7, 'r': 60}, 'ShortPct': {'v': 36.4, 'r': 15}}, 'Swap': {'Net': {'v': 43208.0, 'r': 35}, 'Long': {'v': 57807.0, 'r': 50}, 'Short': {'v': 14599.0, 'r': 72}, 'NetPct': {'v': 12.7, 'r': 7}, 'LongPct': {'v': 16.9, 'r': 1}, 'ShortPct': {'v': 4.3, 'r': 31}}, 'Other_R': {'Net': {'v': 56063.0, 'r': 100}, 'Long': {'v': 78893.0, 'r': 100}, 'Short': {'v': 22830.0, 'r': 88}, 'NetPct': {'v': 16.4, 'r': 100}, 'LongPct': {'v': 23.1, 'r': 100}, 'ShortPct': {'v': 6.7, 'r': 56}}, 'SS': {'Net': {'v': 7497.0, 'r': 68}, 'Long': {'v': 18481.0, 'r': 78}, 'Short': {'v': 10984.0, 'r': 50}, 'NetPct': {'v': 2.2, 'r': 52}, 'LongPct': {'v': 5.4, 'r': 10}, 'ShortPct': {'v': 3.2, 'r': 9}}}, 'Old': {'MM': {'Net': {'v': -42799.0, 'r': 8}, 'Long': {'v': 43835.0, 'r': 57}, 'Short': {'v': 86634.0, 'r': 95}, 'NetPct': {'v': -17.9, 'r': 11}, 'LongPct': {'v': 18.3, 'r': 22}, 'ShortPct': {'v': 36.1, 'r': 90}}, 'Prod': {'Net': {'v': -33971.0, 'r': 69}, 'Long': {'v': 34232.0, 'r': 80}, 'Short': {'v': 68203.0, 'r': 38}, 'NetPct': {'v': -14.2, 'r': 80}, 'LongPct': {'v': 14.3, 'r': 55}, 'ShortPct': {'v': 28.4, 'r': 13}}, 'Swap': {'Net': {'v': 24565.0, 'r': 25}, 'Long': {'v': 32782.0, 'r': 19}, 'Short': {'v': 8217.0, 'r': 50}, 'NetPct': {'v': 10.2, 'r': 10}, 'LongPct': {'v': 13.7, 'r': 5}, 'ShortPct': {'v': 3.4, 'r': 30}}, 'Other_R': {'Net': {'v': 47915.0, 'r': 100}, 'Long': {'v': 74355.0, 'r': 100}, 'Short': {'v': 26440.0, 'r': 93}, 'NetPct': {'v': 20.0, 'r': 100}, 'LongPct': {'v': 31.0, 'r': 100}, 'ShortPct': {'v': 11.0, 'r': 80}}, 'SS': {'Net': {'v': 4290.0, 'r': 61}, 'Long': {'v': 13810.0, 'r': 64}, 'Short': {'v': 9520.0, 'r': 55}, 'NetPct': {'v': 1.8, 'r': 51}, 'LongPct': {'v': 5.8, 'r': 16}, 'ShortPct': {'v': 4.0, 'r': 17}}}, 'Oth': {'MM': {'Net': {'v': 2961.0, 'r': 53}, 'Long': {'v': 17703.0, 'r': 80}, 'Short': {'v': 14742.0, 'r': 94}, 'NetPct': {'v': 2.9, 'r': 26}, 'LongPct': {'v': 17.4, 'r': 51}, 'ShortPct': {'v': 14.5, 'r': 89}}, 'Prod': {'Net': {'v': -32959.0, 'r': 24}, 'Long': {'v': 22920.0, 'r': 95}, 'Short': {'v': 55879.0, 'r': 83}, 'NetPct': {'v': -32.4, 'r': 58}, 'LongPct': {'v': 22.6, 'r': 35}, 'ShortPct': {'v': 55.0, 'r': 15}}, 'Swap': {'Net': {'v': 18643.0, 'r': 76}, 'Long': {'v': 31097.0, 'r': 81}, 'Short': {'v': 12454.0, 'r': 94}, 'NetPct': {'v': 18.4, 'r': 54}, 'LongPct': {'v': 30.6, 'r': 59}, 'ShortPct': {'v': 12.3, 'r': 59}}, 'Other_R': {'Net': {'v': 8148.0, 'r': 83}, 'Long': {'v': 19485.0, 'r': 94}, 'Short': {'v': 11337.0, 'r': 91}, 'NetPct': {'v': 8.0, 'r': 55}, 'LongPct': {'v': 19.2, 'r': 83}, 'ShortPct': {'v': 11.2, 'r': 81}}, 'SS': {'Net': {'v': 3207.0, 'r': 85}, 'Long': {'v': 4671.0, 'r': 75}, 'Short': {'v': 1464.0, 'r': 64}, 'NetPct': {'v': 3.2, 'r': 43}, 'LongPct': {'v': 4.6, 'r': 22}, 'ShortPct': {'v': 1.4, 'r': 13}}}}}, 'Corn': {'OI': 1796424, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 279630.0, 'r': 79}, 'Long': {'v': 363377.0, 'r': 89}, 'Short': {'v': 83747.0, 'r': 51}, 'NetPct': {'v': 15.6, 'r': 64}, 'LongPct': {'v': 20.2, 'r': 65}, 'ShortPct': {'v': 4.7, 'r': 42}}, 'Prod': {'Net': {'v': -563874.0, 'r': 18}, 'Long': {'v': 391778.0, 'r': 70}, 'Short': {'v': 955652.0, 'r': 90}, 'NetPct': {'v': -31.8, 'r': 38}, 'LongPct': {'v': 22.1, 'r': 43}, 'ShortPct': {'v': 53.9, 'r': 67}}, 'Swap': {'Net': {'v': 308251.0, 'r': 82}, 'Long': {'v': 335423.0, 'r': 87}, 'Short': {'v': 27172.0, 'r': 66}, 'NetPct': {'v': 17.4, 'r': 47}, 'LongPct': {'v': 18.9, 'r': 43}, 'ShortPct': {'v': 1.5, 'r': 53}}, 'Other_R': {'Net': {'v': 81454.0, 'r': 56}, 'Long': {'v': 127674.0, 'r': 39}, 'Short': {'v': 46220.0, 'r': 14}, 'NetPct': {'v': 4.6, 'r': 40}, 'LongPct': {'v': 7.2, 'r': 12}, 'ShortPct': {'v': 2.6, 'r': 2}}, 'SS': {'Net': {'v': -56719.0, 'r': 59}, 'Long': {'v': 136940.0, 'r': 45}, 'Short': {'v': 193659.0, 'r': 43}, 'NetPct': {'v': -3.2, 'r': 65}, 'LongPct': {'v': 7.7, 'r': 3}, 'ShortPct': {'v': 10.9, 'r': 15}}}, 'Old': {'MM': {'Net': {'v': 89214.0, 'r': 67}, 'Long': {'v': 255677.0, 'r': 79}, 'Short': {'v': 166463.0, 'r': 63}, 'NetPct': {'v': 7.1, 'r': 59}, 'LongPct': {'v': 20.4, 'r': 56}, 'ShortPct': {'v': 13.3, 'r': 42}}, 'Prod': {'Net': {'v': -388680.0, 'r': 19}, 'Long': {'v': 277496.0, 'r': 74}, 'Short': {'v': 666176.0, 'r': 85}, 'NetPct': {'v': -31.1, 'r': 48}, 'LongPct': {'v': 22.2, 'r': 56}, 'ShortPct': {'v': 53.2, 'r': 62}}, 'Swap': {'Net': {'v': 270950.0, 'r': 85}, 'Long': {'v': 277567.0, 'r': 81}, 'Short': {'v': 6617.0, 'r': 30}, 'NetPct': {'v': 21.7, 'r': 51}, 'LongPct': {'v': 22.2, 'r': 46}, 'ShortPct': {'v': 0.5, 'r': 17}}, 'Other_R': {'Net': {'v': 35307.0, 'r': 45}, 'Long': {'v': 132839.0, 'r': 64}, 'Short': {'v': 97532.0, 'r': 85}, 'NetPct': {'v': 2.8, 'r': 28}, 'LongPct': {'v': 10.6, 'r': 33}, 'ShortPct': {'v': 7.8, 'r': 61}}, 'SS': {'Net': {'v': -6791.0, 'r': 69}, 'Long': {'v': 107376.0, 'r': 56}, 'Short': {'v': 114167.0, 'r': 39}, 'NetPct': {'v': -0.5, 'r': 71}, 'LongPct': {'v': 8.6, 'r': 19}, 'ShortPct': {'v': 9.1, 'r': 15}}}, 'Oth': {'MM': {'Net': {'v': 141674.0, 'r': 86}, 'Long': {'v': 202915.0, 'r': 90}, 'Short': {'v': 61241.0, 'r': 77}, 'NetPct': {'v': 27.1, 'r': 84}, 'LongPct': {'v': 38.9, 'r': 97}, 'ShortPct': {'v': 11.7, 'r': 61}}, 'Prod': {'Net': {'v': -175194.0, 'r': 23}, 'Long': {'v': 114282.0, 'r': 55}, 'Short': {'v': 289476.0, 'r': 72}, 'NetPct': {'v': -33.5, 'r': 23}, 'LongPct': {'v': 21.9, 'r': 20}, 'ShortPct': {'v': 55.4, 'r': 73}}, 'Swap': {'Net': {'v': 37301.0, 'r': 56}, 'Long': {'v': 59984.0, 'r': 57}, 'Short': {'v': 22683.0, 'r': 66}, 'NetPct': {'v': 7.1, 'r': 34}, 'LongPct': {'v': 11.5, 'r': 26}, 'ShortPct': {'v': 4.3, 'r': 55}}, 'Other_R': {'Net': {'v': 46147.0, 'r': 68}, 'Long': {'v': 85817.0, 'r': 69}, 'Short': {'v': 39670.0, 'r': 66}, 'NetPct': {'v': 8.8, 'r': 63}, 'LongPct': {'v': 16.4, 'r': 67}, 'ShortPct': {'v': 7.6, 'r': 60}}, 'SS': {'Net': {'v': -49928.0, 'r': 29}, 'Long': {'v': 29564.0, 'r': 44}, 'Short': {'v': 79492.0, 'r': 55}, 'NetPct': {'v': -9.6, 'r': 43}, 'LongPct': {'v': 5.7, 'r': 4}, 'ShortPct': {'v': 15.2, 'r': 22}}}}}, 'Soybeans': {'OI': 959188, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 191350.0, 'r': 96}, 'Long': {'v': 227533.0, 'r': 98}, 'Short': {'v': 36183.0, 'r': 54}, 'NetPct': {'v': 19.9, 'r': 77}, 'LongPct': {'v': 23.7, 'r': 83}, 'ShortPct': {'v': 3.8, 'r': 34}}, 'Prod': {'Net': {'v': -296896.0, 'r': 10}, 'Long': {'v': 262475.0, 'r': 78}, 'Short': {'v': 559371.0, 'r': 97}, 'NetPct': {'v': -30.3, 'r': 42}, 'LongPct': {'v': 26.8, 'r': 50}, 'ShortPct': {'v': 57.1, 'r': 67}}, 'Swap': {'Net': {'v': 106395.0, 'r': 48}, 'Long': {'v': 158086.0, 'r': 95}, 'Short': {'v': 51691.0, 'r': 100}, 'NetPct': {'v': 10.9, 'r': 9}, 'LongPct': {'v': 16.1, 'r': 40}, 'ShortPct': {'v': 5.3, 'r': 95}}, 'Other_R': {'Net': {'v': 25845.0, 'r': 62}, 'Long': {'v': 59618.0, 'r': 49}, 'Short': {'v': 33773.0, 'r': 45}, 'NetPct': {'v': 2.6, 'r': 47}, 'LongPct': {'v': 6.1, 'r': 14}, 'ShortPct': {'v': 3.4, 'r': 10}}, 'SS': {'Net': {'v': -30565.0, 'r': 41}, 'Long': {'v': 50469.0, 'r': 46}, 'Short': {'v': 81034.0, 'r': 55}, 'NetPct': {'v': -3.1, 'r': 68}, 'LongPct': {'v': 5.1, 'r': 3}, 'ShortPct': {'v': 8.3, 'r': 14}}}, 'Old': {'MM': {'Net': {'v': 124188.0, 'r': 90}, 'Long': {'v': 187797.0, 'r': 97}, 'Short': {'v': 63609.0, 'r': 70}, 'NetPct': {'v': 18.7, 'r': 72}, 'LongPct': {'v': 28.3, 'r': 87}, 'ShortPct': {'v': 9.6, 'r': 56}}, 'Prod': {'Net': {'v': -221174.0, 'r': 14}, 'Long': {'v': 170788.0, 'r': 66}, 'Short': {'v': 391962.0, 'r': 92}, 'NetPct': {'v': -33.3, 'r': 43}, 'LongPct': {'v': 25.7, 'r': 46}, 'ShortPct': {'v': 59.0, 'r': 60}}, 'Swap': {'Net': {'v': 74357.0, 'r': 38}, 'Long': {'v': 110187.0, 'r': 62}, 'Short': {'v': 35830.0, 'r': 96}, 'NetPct': {'v': 11.2, 'r': 20}, 'LongPct': {'v': 16.6, 'r': 40}, 'ShortPct': {'v': 5.4, 'r': 92}}, 'Other_R': {'Net': {'v': 34145.0, 'r': 81}, 'Long': {'v': 62445.0, 'r': 69}, 'Short': {'v': 28300.0, 'r': 51}, 'NetPct': {'v': 5.1, 'r': 61}, 'LongPct': {'v': 9.4, 'r': 36}, 'ShortPct': {'v': 4.3, 'r': 13}}, 'SS': {'Net': {'v': -11516.0, 'r': 53}, 'Long': {'v': 34675.0, 'r': 35}, 'Short': {'v': 46191.0, 'r': 36}, 'NetPct': {'v': -1.7, 'r': 65}, 'LongPct': {'v': 5.2, 'r': 4}, 'ShortPct': {'v': 7.0, 'r': 2}}}, 'Oth': {'MM': {'Net': {'v': 71033.0, 'r': 92}, 'Long': {'v': 86808.0, 'r': 90}, 'Short': {'v': 15775.0, 'r': 63}, 'NetPct': {'v': 22.5, 'r': 73}, 'LongPct': {'v': 27.5, 'r': 66}, 'ShortPct': {'v': 5.0, 'r': 28}}, 'Prod': {'Net': {'v': -75722.0, 'r': 17}, 'Long': {'v': 91687.0, 'r': 77}, 'Short': {'v': 167409.0, 'r': 80}, 'NetPct': {'v': -24.0, 'r': 32}, 'LongPct': {'v': 29.0, 'r': 43}, 'ShortPct': {'v': 53.0, 'r': 72}}, 'Swap': {'Net': {'v': 32038.0, 'r': 73}, 'Long': {'v': 50044.0, 'r': 78}, 'Short': {'v': 18006.0, 'r': 90}, 'NetPct': {'v': 10.1, 'r': 47}, 'LongPct': {'v': 15.8, 'r': 57}, 'ShortPct': {'v': 5.7, 'r': 75}}, 'Other_R': {'Net': {'v': -8300.0, 'r': 13}, 'Long': {'v': 16940.0, 'r': 56}, 'Short': {'v': 25240.0, 'r': 73}, 'NetPct': {'v': -2.6, 'r': 33}, 'LongPct': {'v': 5.4, 'r': 10}, 'ShortPct': {'v': 8.0, 'r': 40}}, 'SS': {'Net': {'v': -19049.0, 'r': 31}, 'Long': {'v': 15794.0, 'r': 64}, 'Short': {'v': 34843.0, 'r': 64}, 'NetPct': {'v': -6.0, 'r': 66}, 'LongPct': {'v': 5.0, 'r': 1}, 'ShortPct': {'v': 11.0, 'r': 18}}}}}, 'SRW Wheat': {'OI': 484568, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': -994.0, 'r': 64}, 'Long': {'v': 110732.0, 'r': 93}, 'Short': {'v': 111726.0, 'r': 56}, 'NetPct': {'v': -0.2, 'r': 66}, 'LongPct': {'v': 22.9, 'r': 68}, 'ShortPct': {'v': 23.1, 'r': 41}}, 'Prod': {'Net': {'v': -49804.0, 'r': 58}, 'Long': {'v': 47416.0, 'r': 48}, 'Short': {'v': 97220.0, 'r': 37}, 'NetPct': {'v': -10.4, 'r': 62}, 'LongPct': {'v': 9.9, 'r': 31}, 'ShortPct': {'v': 20.3, 'r': 24}}, 'Swap': {'Net': {'v': 74393.0, 'r': 45}, 'Long': {'v': 93973.0, 'r': 53}, 'Short': {'v': 19580.0, 'r': 84}, 'NetPct': {'v': 15.5, 'r': 24}, 'LongPct': {'v': 19.6, 'r': 40}, 'ShortPct': {'v': 4.1, 'r': 77}}, 'Other_R': {'Net': {'v': -13794.0, 'r': 3}, 'Long': {'v': 29253.0, 'r': 12}, 'Short': {'v': 43047.0, 'r': 90}, 'NetPct': {'v': -2.9, 'r': 5}, 'LongPct': {'v': 6.1, 'r': 2}, 'ShortPct': {'v': 9.0, 'r': 83}}, 'SS': {'Net': {'v': 1079.0, 'r': 90}, 'Long': {'v': 33774.0, 'r': 57}, 'Short': {'v': 32695.0, 'r': 6}, 'NetPct': {'v': 0.2, 'r': 90}, 'LongPct': {'v': 7.1, 'r': 16}, 'ShortPct': {'v': 6.8, 'r': 4}}}, 'Old': {'MM': {'Net': {'v': -23141.0, 'r': 50}, 'Long': {'v': 93045.0, 'r': 86}, 'Short': {'v': 116186.0, 'r': 70}, 'NetPct': {'v': -10.2, 'r': 43}, 'LongPct': {'v': 40.8, 'r': 97}, 'ShortPct': {'v': 51.0, 'r': 94}}, 'Prod': {'Net': {'v': 3744.0, 'r': 83}, 'Long': {'v': 24966.0, 'r': 26}, 'Short': {'v': 21222.0, 'r': 6}, 'NetPct': {'v': 1.6, 'r': 83}, 'LongPct': {'v': 11.0, 'r': 40}, 'ShortPct': {'v': 9.3, 'r': 2}}, 'Swap': {'Net': {'v': 12417.0, 'r': 10}, 'Long': {'v': 34614.0, 'r': 15}, 'Short': {'v': 22197.0, 'r': 95}, 'NetPct': {'v': 5.5, 'r': 10}, 'LongPct': {'v': 15.2, 'r': 27}, 'ShortPct': {'v': 9.7, 'r': 97}}, 'Other_R': {'Net': {'v': 2517.0, 'r': 23}, 'Long': {'v': 32354.0, 'r': 25}, 'Short': {'v': 29837.0, 'r': 57}, 'NetPct': {'v': 1.1, 'r': 24}, 'LongPct': {'v': 14.2, 'r': 56}, 'ShortPct': {'v': 13.1, 'r': 82}}, 'SS': {'Net': {'v': 4463.0, 'r': 93}, 'Long': {'v': 18389.0, 'r': 14}, 'Short': {'v': 13926.0, 'r': 8}, 'NetPct': {'v': 2.0, 'r': 95}, 'LongPct': {'v': 8.1, 'r': 44}, 'ShortPct': {'v': 6.1, 'r': 1}}}, 'Oth': {'MM': {'Net': {'v': 11267.0, 'r': 84}, 'Long': {'v': 78264.0, 'r': 97}, 'Short': {'v': 66997.0, 'r': 89}, 'NetPct': {'v': 4.5, 'r': 58}, 'LongPct': {'v': 31.2, 'r': 77}, 'ShortPct': {'v': 26.7, 'r': 66}}, 'Prod': {'Net': {'v': -53548.0, 'r': 7}, 'Long': {'v': 22450.0, 'r': 83}, 'Short': {'v': 75998.0, 'r': 93}, 'NetPct': {'v': -21.4, 'r': 41}, 'LongPct': {'v': 9.0, 'r': 31}, 'ShortPct': {'v': 30.3, 'r': 49}}, 'Swap': {'Net': {'v': 61976.0, 'r': 92}, 'Long': {'v': 69906.0, 'r': 94}, 'Short': {'v': 7930.0, 'r': 82}, 'NetPct': {'v': 24.7, 'r': 67}, 'LongPct': {'v': 27.9, 'r': 64}, 'ShortPct': {'v': 3.2, 'r': 41}}, 'Other_R': {'Net': {'v': -16311.0, 'r': 1}, 'Long': {'v': 11738.0, 'r': 52}, 'Short': {'v': 28049.0, 'r': 95}, 'NetPct': {'v': -6.5, 'r': 12}, 'LongPct': {'v': 4.7, 'r': 14}, 'ShortPct': {'v': 11.2, 'r': 71}}, 'SS': {'Net': {'v': -3384.0, 'r': 51}, 'Long': {'v': 15385.0, 'r': 83}, 'Short': {'v': 18769.0, 'r': 78}, 'NetPct': {'v': -1.4, 'r': 88}, 'LongPct': {'v': 6.1, 'r': 17}, 'ShortPct': {'v': 7.5, 'r': 8}}}}}, 'Sugar': {'OI': 940662, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': -73170.0, 'r': 2}, 'Long': {'v': 149710.0, 'r': 42}, 'Short': {'v': 222880.0, 'r': 99}, 'NetPct': {'v': -7.8, 'r': 2}, 'LongPct': {'v': 15.9, 'r': 11}, 'ShortPct': {'v': 23.7, 'r': 99}}, 'Prod': {'Net': {'v': 9053.0, 'r': 95}, 'Long': {'v': 265045.0, 'r': 69}, 'Short': {'v': 255992.0, 'r': 3}, 'NetPct': {'v': 0.9, 'r': 95}, 'LongPct': {'v': 26.4, 'r': 45}, 'ShortPct': {'v': 25.5, 'r': 1}}, 'Swap': {'Net': {'v': 191015.0, 'r': 92}, 'Long': {'v': 239117.0, 'r': 92}, 'Short': {'v': 48102.0, 'r': 31}, 'NetPct': {'v': 19.0, 'r': 70}, 'LongPct': {'v': 23.8, 'r': 62}, 'ShortPct': {'v': 4.8, 'r': 21}}, 'Other_R': {'Net': {'v': -9616.0, 'r': 5}, 'Long': {'v': 55695.0, 'r': 48}, 'Short': {'v': 65311.0, 'r': 96}, 'NetPct': {'v': -1.0, 'r': 6}, 'LongPct': {'v': 5.5, 'r': 28}, 'ShortPct': {'v': 6.5, 'r': 95}}, 'SS': {'Net': {'v': 7687.0, 'r': 43}, 'Long': {'v': 85031.0, 'r': 78}, 'Short': {'v': 77344.0, 'r': 86}, 'NetPct': {'v': 0.8, 'r': 40}, 'LongPct': {'v': 8.5, 'r': 30}, 'ShortPct': {'v': 7.7, 'r': 49}}}, 'Old': {'MM': {'Net': {'v': -198139.0, 'r': 2}, 'Long': {'v': 136614.0, 'r': 42}, 'Short': {'v': 334753.0, 'r': 99}, 'NetPct': {'v': -19.7, 'r': 2}, 'LongPct': {'v': 13.6, 'r': 11}, 'ShortPct': {'v': 33.3, 'r': 99}}, 'Prod': {'Net': {'v': 9053.0, 'r': 95}, 'Long': {'v': 265045.0, 'r': 69}, 'Short': {'v': 255992.0, 'r': 3}, 'NetPct': {'v': 0.9, 'r': 95}, 'LongPct': {'v': 26.4, 'r': 45}, 'ShortPct': {'v': 25.5, 'r': 1}}, 'Swap': {'Net': {'v': 191015.0, 'r': 92}, 'Long': {'v': 239117.0, 'r': 92}, 'Short': {'v': 48102.0, 'r': 31}, 'NetPct': {'v': 19.0, 'r': 70}, 'LongPct': {'v': 23.8, 'r': 62}, 'ShortPct': {'v': 4.8, 'r': 21}}, 'Other_R': {'Net': {'v': -9616.0, 'r': 5}, 'Long': {'v': 55695.0, 'r': 48}, 'Short': {'v': 65311.0, 'r': 96}, 'NetPct': {'v': -1.0, 'r': 6}, 'LongPct': {'v': 5.5, 'r': 28}, 'ShortPct': {'v': 6.5, 'r': 95}}, 'SS': {'Net': {'v': 7687.0, 'r': 43}, 'Long': {'v': 85031.0, 'r': 78}, 'Short': {'v': 77344.0, 'r': 86}, 'NetPct': {'v': 0.8, 'r': 40}, 'LongPct': {'v': 8.5, 'r': 30}, 'ShortPct': {'v': 7.7, 'r': 49}}}, 'Oth': {'MM': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Prod': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Swap': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Other_R': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'SS': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}}}}, 'WTI Crude': {'OI': 2093735, 'date': '2022-02-01', 'current': {'ALL': {'MM': {'Net': {'v': 269135.0, 'r': 77}, 'Long': {'v': 312271.0, 'r': 72}, 'Short': {'v': 43136.0, 'r': 20}, 'NetPct': {'v': 12.9, 'r': 66}, 'LongPct': {'v': 14.9, 'r': 52}, 'ShortPct': {'v': 2.1, 'r': 14}}, 'Prod': {'Net': {'v': 49956.0, 'r': 99}, 'Long': {'v': 490091.0, 'r': 94}, 'Short': {'v': 440135.0, 'r': 67}, 'NetPct': {'v': 2.4, 'r': 98}, 'LongPct': {'v': 23.4, 'r': 89}, 'ShortPct': {'v': 21.0, 'r': 37}}, 'Swap': {'Net': {'v': -464614.0, 'r': 14}, 'Long': {'v': 107420.0, 'r': 1}, 'Short': {'v': 572034.0, 'r': 79}, 'NetPct': {'v': -22.2, 'r': 13}, 'LongPct': {'v': 5.1, 'r': 0}, 'ShortPct': {'v': 27.3, 'r': 76}}, 'Other_R': {'Net': {'v': 99769.0, 'r': 61}, 'Long': {'v': 175335.0, 'r': 58}, 'Short': {'v': 75566.0, 'r': 25}, 'NetPct': {'v': 4.8, 'r': 51}, 'LongPct': {'v': 8.4, 'r': 42}, 'ShortPct': {'v': 3.6, 'r': 16}}, 'SS': {'Net': {'v': 45754.0, 'r': 95}, 'Long': {'v': 99747.0, 'r': 78}, 'Short': {'v': 53993.0, 'r': 7}, 'NetPct': {'v': 2.2, 'r': 93}, 'LongPct': {'v': 4.8, 'r': 40}, 'ShortPct': {'v': 2.6, 'r': 7}}}, 'Old': {'MM': {'Net': {'v': 269135.0, 'r': 77}, 'Long': {'v': 312271.0, 'r': 72}, 'Short': {'v': 43136.0, 'r': 20}, 'NetPct': {'v': 12.9, 'r': 66}, 'LongPct': {'v': 14.9, 'r': 52}, 'ShortPct': {'v': 2.1, 'r': 14}}, 'Prod': {'Net': {'v': 49956.0, 'r': 99}, 'Long': {'v': 490091.0, 'r': 94}, 'Short': {'v': 440135.0, 'r': 67}, 'NetPct': {'v': 2.4, 'r': 98}, 'LongPct': {'v': 23.4, 'r': 89}, 'ShortPct': {'v': 21.0, 'r': 37}}, 'Swap': {'Net': {'v': -464614.0, 'r': 14}, 'Long': {'v': 107420.0, 'r': 1}, 'Short': {'v': 572034.0, 'r': 79}, 'NetPct': {'v': -22.2, 'r': 13}, 'LongPct': {'v': 5.1, 'r': 0}, 'ShortPct': {'v': 27.3, 'r': 76}}, 'Other_R': {'Net': {'v': 99769.0, 'r': 61}, 'Long': {'v': 175335.0, 'r': 58}, 'Short': {'v': 75566.0, 'r': 25}, 'NetPct': {'v': 4.8, 'r': 51}, 'LongPct': {'v': 8.4, 'r': 42}, 'ShortPct': {'v': 3.6, 'r': 16}}, 'SS': {'Net': {'v': 45754.0, 'r': 95}, 'Long': {'v': 99747.0, 'r': 78}, 'Short': {'v': 53993.0, 'r': 7}, 'NetPct': {'v': 2.2, 'r': 93}, 'LongPct': {'v': 4.8, 'r': 40}, 'ShortPct': {'v': 2.6, 'r': 7}}}, 'Oth': {'MM': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Prod': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Swap': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Other_R': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'SS': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}}}}, 'Gold': {'OI': 403925, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 91621.0, 'r': 49}, 'Long': {'v': 119562.0, 'r': 44}, 'Short': {'v': 27941.0, 'r': 44}, 'NetPct': {'v': 22.7, 'r': 52}, 'LongPct': {'v': 29.6, 'r': 56}, 'ShortPct': {'v': 6.9, 'r': 48}}, 'Prod': {'Net': {'v': -17834.0, 'r': 96}, 'Long': {'v': 15879.0, 'r': 18}, 'Short': {'v': 33713.0, 'r': 1}, 'NetPct': {'v': -4.3, 'r': 96}, 'LongPct': {'v': 3.9, 'r': 30}, 'ShortPct': {'v': 8.2, 'r': 1}}, 'Swap': {'Net': {'v': -180814.0, 'r': 17}, 'Long': {'v': 36975.0, 'r': 18}, 'Short': {'v': 217789.0, 'r': 73}, 'NetPct': {'v': -44.0, 'r': 5}, 'LongPct': {'v': 9.0, 'r': 18}, 'ShortPct': {'v': 52.9, 'r': 94}}, 'Other_R': {'Net': {'v': 57826.0, 'r': 60}, 'Long': {'v': 85814.0, 'r': 62}, 'Short': {'v': 27988.0, 'r': 53}, 'NetPct': {'v': 14.1, 'r': 68}, 'LongPct': {'v': 20.9, 'r': 72}, 'ShortPct': {'v': 6.8, 'r': 67}}, 'SS': {'Net': {'v': 38779.0, 'r': 80}, 'Long': {'v': 52913.0, 'r': 56}, 'Short': {'v': 14134.0, 'r': 2}, 'NetPct': {'v': 9.4, 'r': 90}, 'LongPct': {'v': 12.9, 'r': 75}, 'ShortPct': {'v': 3.4, 'r': 4}}}, 'Old': {'MM': {'Net': {'v': 102043.0, 'r': 49}, 'Long': {'v': 130147.0, 'r': 44}, 'Short': {'v': 28104.0, 'r': 44}, 'NetPct': {'v': 24.8, 'r': 52}, 'LongPct': {'v': 31.6, 'r': 56}, 'ShortPct': {'v': 6.8, 'r': 48}}, 'Prod': {'Net': {'v': -17834.0, 'r': 96}, 'Long': {'v': 15879.0, 'r': 18}, 'Short': {'v': 33713.0, 'r': 1}, 'NetPct': {'v': -4.3, 'r': 96}, 'LongPct': {'v': 3.9, 'r': 30}, 'ShortPct': {'v': 8.2, 'r': 1}}, 'Swap': {'Net': {'v': -180814.0, 'r': 17}, 'Long': {'v': 36975.0, 'r': 18}, 'Short': {'v': 217789.0, 'r': 73}, 'NetPct': {'v': -44.0, 'r': 5}, 'LongPct': {'v': 9.0, 'r': 18}, 'ShortPct': {'v': 52.9, 'r': 94}}, 'Other_R': {'Net': {'v': 57826.0, 'r': 60}, 'Long': {'v': 85814.0, 'r': 62}, 'Short': {'v': 27988.0, 'r': 53}, 'NetPct': {'v': 14.1, 'r': 68}, 'LongPct': {'v': 20.9, 'r': 72}, 'ShortPct': {'v': 6.8, 'r': 67}}, 'SS': {'Net': {'v': 38779.0, 'r': 80}, 'Long': {'v': 52913.0, 'r': 56}, 'Short': {'v': 14134.0, 'r': 2}, 'NetPct': {'v': 9.4, 'r': 90}, 'LongPct': {'v': 12.9, 'r': 75}, 'ShortPct': {'v': 3.4, 'r': 4}}}, 'Oth': {'MM': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Prod': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Swap': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Other_R': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'SS': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}}}}, 'Silver': {'OI': 113164, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 11158.0, 'r': 25}, 'Long': {'v': 15246.0, 'r': 0}, 'Short': {'v': 4088.0, 'r': 8}, 'NetPct': {'v': 9.9, 'r': 27}, 'LongPct': {'v': 13.5, 'r': 1}, 'ShortPct': {'v': 3.6, 'r': 9}}, 'Prod': {'Net': {'v': -14797.0, 'r': 100}, 'Long': {'v': 3847.0, 'r': 6}, 'Short': {'v': 18644.0, 'r': 0}, 'NetPct': {'v': -12.9, 'r': 93}, 'LongPct': {'v': 3.4, 'r': 13}, 'ShortPct': {'v': 16.2, 'r': 1}}, 'Swap': {'Net': {'v': -23561.0, 'r': 17}, 'Long': {'v': 21883.0, 'r': 34}, 'Short': {'v': 45444.0, 'r': 66}, 'NetPct': {'v': -20.5, 'r': 8}, 'LongPct': {'v': 19.1, 'r': 53}, 'ShortPct': {'v': 39.6, 'r': 93}}, 'Other_R': {'Net': {'v': 12234.0, 'r': 65}, 'Long': {'v': 18357.0, 'r': 65}, 'Short': {'v': 6123.0, 'r': 58}, 'NetPct': {'v': 10.7, 'r': 83}, 'LongPct': {'v': 16.0, 'r': 93}, 'ShortPct': {'v': 5.3, 'r': 83}}, 'SS': {'Net': {'v': 16477.0, 'r': 56}, 'Long': {'v': 26297.0, 'r': 44}, 'Short': {'v': 9820.0, 'r': 24}, 'NetPct': {'v': 14.4, 'r': 85}, 'LongPct': {'v': 22.9, 'r': 86}, 'ShortPct': {'v': 8.6, 'r': 61}}}, 'Old': {'MM': {'Net': {'v': 9647.0, 'r': 25}, 'Long': {'v': 12768.0, 'r': 0}, 'Short': {'v': 3121.0, 'r': 8}, 'NetPct': {'v': 8.4, 'r': 27}, 'LongPct': {'v': 11.1, 'r': 1}, 'ShortPct': {'v': 2.7, 'r': 9}}, 'Prod': {'Net': {'v': -14797.0, 'r': 100}, 'Long': {'v': 3847.0, 'r': 6}, 'Short': {'v': 18644.0, 'r': 0}, 'NetPct': {'v': -12.9, 'r': 93}, 'LongPct': {'v': 3.4, 'r': 13}, 'ShortPct': {'v': 16.2, 'r': 1}}, 'Swap': {'Net': {'v': -23561.0, 'r': 17}, 'Long': {'v': 21883.0, 'r': 34}, 'Short': {'v': 45444.0, 'r': 66}, 'NetPct': {'v': -20.5, 'r': 8}, 'LongPct': {'v': 19.1, 'r': 53}, 'ShortPct': {'v': 39.6, 'r': 93}}, 'Other_R': {'Net': {'v': 12234.0, 'r': 65}, 'Long': {'v': 18357.0, 'r': 65}, 'Short': {'v': 6123.0, 'r': 58}, 'NetPct': {'v': 10.7, 'r': 83}, 'LongPct': {'v': 16.0, 'r': 93}, 'ShortPct': {'v': 5.3, 'r': 83}}, 'SS': {'Net': {'v': 16477.0, 'r': 56}, 'Long': {'v': 26297.0, 'r': 44}, 'Short': {'v': 9820.0, 'r': 24}, 'NetPct': {'v': 14.4, 'r': 85}, 'LongPct': {'v': 22.9, 'r': 86}, 'ShortPct': {'v': 8.6, 'r': 61}}}, 'Oth': {'MM': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Prod': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Swap': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Other_R': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'SS': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}}}}, 'Live Cattle': {'OI': 334831, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 107593.0, 'r': 82}, 'Long': {'v': 122119.0, 'r': 82}, 'Short': {'v': 14526.0, 'r': 19}, 'NetPct': {'v': 32.1, 'r': 84}, 'LongPct': {'v': 36.5, 'r': 85}, 'ShortPct': {'v': 4.3, 'r': 19}}, 'Prod': {'Net': {'v': -123957.0, 'r': 44}, 'Long': {'v': 40397.0, 'r': 69}, 'Short': {'v': 164354.0, 'r': 62}, 'NetPct': {'v': -37.1, 'r': 50}, 'LongPct': {'v': 12.1, 'r': 60}, 'ShortPct': {'v': 49.2, 'r': 53}}, 'Swap': {'Net': {'v': 58686.0, 'r': 15}, 'Long': {'v': 64215.0, 'r': 17}, 'Short': {'v': 5529.0, 'r': 85}, 'NetPct': {'v': 17.6, 'r': 9}, 'LongPct': {'v': 19.2, 'r': 10}, 'ShortPct': {'v': 1.7, 'r': 83}}, 'Other_R': {'Net': {'v': -24981.0, 'r': 4}, 'Long': {'v': 15585.0, 'r': 30}, 'Short': {'v': 40566.0, 'r': 84}, 'NetPct': {'v': -7.5, 'r': 6}, 'LongPct': {'v': 4.7, 'r': 23}, 'ShortPct': {'v': 12.1, 'r': 79}}, 'SS': {'Net': {'v': -12891.0, 'r': 64}, 'Long': {'v': 29556.0, 'r': 59}, 'Short': {'v': 42447.0, 'r': 37}, 'NetPct': {'v': -3.9, 'r': 70}, 'LongPct': {'v': 8.8, 'r': 45}, 'ShortPct': {'v': 12.7, 'r': 21}}}, 'Old': {'MM': {'Net': {'v': 103143.0, 'r': 82}, 'Long': {'v': 117445.0, 'r': 82}, 'Short': {'v': 14302.0, 'r': 19}, 'NetPct': {'v': 30.9, 'r': 84}, 'LongPct': {'v': 35.1, 'r': 85}, 'ShortPct': {'v': 4.3, 'r': 19}}, 'Prod': {'Net': {'v': -123957.0, 'r': 44}, 'Long': {'v': 40397.0, 'r': 69}, 'Short': {'v': 164354.0, 'r': 62}, 'NetPct': {'v': -37.1, 'r': 50}, 'LongPct': {'v': 12.1, 'r': 60}, 'ShortPct': {'v': 49.2, 'r': 53}}, 'Swap': {'Net': {'v': 58686.0, 'r': 15}, 'Long': {'v': 64215.0, 'r': 17}, 'Short': {'v': 5529.0, 'r': 85}, 'NetPct': {'v': 17.6, 'r': 9}, 'LongPct': {'v': 19.2, 'r': 10}, 'ShortPct': {'v': 1.7, 'r': 83}}, 'Other_R': {'Net': {'v': -24981.0, 'r': 4}, 'Long': {'v': 15585.0, 'r': 30}, 'Short': {'v': 40566.0, 'r': 84}, 'NetPct': {'v': -7.5, 'r': 6}, 'LongPct': {'v': 4.7, 'r': 23}, 'ShortPct': {'v': 12.1, 'r': 79}}, 'SS': {'Net': {'v': -12891.0, 'r': 64}, 'Long': {'v': 29556.0, 'r': 59}, 'Short': {'v': 42447.0, 'r': 37}, 'NetPct': {'v': -3.9, 'r': 70}, 'LongPct': {'v': 8.8, 'r': 45}, 'ShortPct': {'v': 12.7, 'r': 21}}}, 'Oth': {'MM': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Prod': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Swap': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'Other_R': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}, 'SS': {'Net': {'v': 0.0, 'r': 100}, 'Long': {'v': 0.0, 'r': 100}, 'Short': {'v': 0.0, 'r': 100}, 'NetPct': {'v': 0, 'r': 50}, 'LongPct': {'v': 0, 'r': 50}, 'ShortPct': {'v': 0, 'r': 50}}}}}, 'Lean Hogs': {'OI': 337624, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 93770.0, 'r': 97}, 'Long': {'v': 114103.0, 'r': 98}, 'Short': {'v': 20333.0, 'r': 50}, 'NetPct': {'v': 27.8, 'r': 95}, 'LongPct': {'v': 33.8, 'r': 97}, 'ShortPct': {'v': 6.0, 'r': 23}}, 'Prod': {'Net': {'v': -124114.0, 'r': 9}, 'Long': {'v': 28980.0, 'r': 88}, 'Short': {'v': 153094.0, 'r': 95}, 'NetPct': {'v': -34.5, 'r': 55}, 'LongPct': {'v': 8.1, 'r': 60}, 'ShortPct': {'v': 42.6, 'r': 44}}, 'Swap': {'Net': {'v': 70215.0, 'r': 77}, 'Long': {'v': 81508.0, 'r': 92}, 'Short': {'v': 11293.0, 'r': 96}, 'NetPct': {'v': 19.5, 'r': 15}, 'LongPct': {'v': 22.7, 'r': 20}, 'ShortPct': {'v': 3.1, 'r': 91}}, 'Other_R': {'Net': {'v': -46678.0, 'r': 1}, 'Long': {'v': 14825.0, 'r': 40}, 'Short': {'v': 61503.0, 'r': 95}, 'NetPct': {'v': -13.0, 'r': 3}, 'LongPct': {'v': 4.1, 'r': 14}, 'ShortPct': {'v': 17.1, 'r': 88}}, 'SS': {'Net': {'v': -9159.0, 'r': 42}, 'Long': {'v': 20617.0, 'r': 38}, 'Short': {'v': 29776.0, 'r': 38}, 'NetPct': {'v': -2.5, 'r': 68}, 'LongPct': {'v': 5.7, 'r': 1}, 'ShortPct': {'v': 8.3, 'r': 4}}}, 'Old': {'MM': {'Net': {'v': 92983.0, 'r': 97}, 'Long': {'v': 118720.0, 'r': 98}, 'Short': {'v': 25737.0, 'r': 56}, 'NetPct': {'v': 29.6, 'r': 95}, 'LongPct': {'v': 37.8, 'r': 86}, 'ShortPct': {'v': 8.2, 'r': 26}}, 'Prod': {'Net': {'v': -111672.0, 'r': 7}, 'Long': {'v': 22098.0, 'r': 86}, 'Short': {'v': 133770.0, 'r': 95}, 'NetPct': {'v': -35.6, 'r': 40}, 'LongPct': {'v': 7.0, 'r': 57}, 'ShortPct': {'v': 42.6, 'r': 60}}, 'Swap': {'Net': {'v': 65115.0, 'r': 79}, 'Long': {'v': 75703.0, 'r': 92}, 'Short': {'v': 10588.0, 'r': 98}, 'NetPct': {'v': 20.8, 'r': 22}, 'LongPct': {'v': 24.1, 'r': 28}, 'ShortPct': {'v': 3.4, 'r': 85}}, 'Other_R': {'Net': {'v': -35987.0, 'r': 2}, 'Long': {'v': 13805.0, 'r': 43}, 'Short': {'v': 49792.0, 'r': 96}, 'NetPct': {'v': -11.5, 'r': 9}, 'LongPct': {'v': 4.4, 'r': 11}, 'ShortPct': {'v': 15.9, 'r': 72}}, 'SS': {'Net': {'v': -10439.0, 'r': 24}, 'Long': {'v': 15330.0, 'r': 36}, 'Short': {'v': 25769.0, 'r': 50}, 'NetPct': {'v': -3.3, 'r': 60}, 'LongPct': {'v': 4.9, 'r': 1}, 'ShortPct': {'v': 8.2, 'r': 5}}}, 'Oth': {'MM': {'Net': {'v': 16753.0, 'r': 77}, 'Long': {'v': 19984.0, 'r': 72}, 'Short': {'v': 3231.0, 'r': 75}, 'NetPct': {'v': 36.8, 'r': 85}, 'LongPct': {'v': 44.0, 'r': 91}, 'ShortPct': {'v': 7.1, 'r': 81}}, 'Prod': {'Net': {'v': -12442.0, 'r': 45}, 'Long': {'v': 6882.0, 'r': 67}, 'Short': {'v': 19324.0, 'r': 57}, 'NetPct': {'v': -27.4, 'r': 72}, 'LongPct': {'v': 15.1, 'r': 57}, 'ShortPct': {'v': 42.5, 'r': 23}}, 'Swap': {'Net': {'v': 5100.0, 'r': 60}, 'Long': {'v': 6458.0, 'r': 60}, 'Short': {'v': 1358.0, 'r': 74}, 'NetPct': {'v': 11.2, 'r': 53}, 'LongPct': {'v': 14.2, 'r': 54}, 'ShortPct': {'v': 3.0, 'r': 75}}, 'Other_R': {'Net': {'v': -10691.0, 'r': 10}, 'Long': {'v': 3613.0, 'r': 54}, 'Short': {'v': 14304.0, 'r': 83}, 'NetPct': {'v': -23.5, 'r': 9}, 'LongPct': {'v': 7.9, 'r': 24}, 'ShortPct': {'v': 31.5, 'r': 89}}, 'SS': {'Net': {'v': 1280.0, 'r': 90}, 'Long': {'v': 5287.0, 'r': 56}, 'Short': {'v': 4007.0, 'r': 52}, 'NetPct': {'v': 2.8, 'r': 60}, 'LongPct': {'v': 11.6, 'r': 27}, 'ShortPct': {'v': 8.8, 'r': 6}}}}}, 'Soy Oil': {'OI': 738609, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 118496.0, 'r': 100}, 'Long': {'v': 148285.0, 'r': 100}, 'Short': {'v': 29789.0, 'r': 47}, 'NetPct': {'v': 16.0, 'r': 81}, 'LongPct': {'v': 20.1, 'r': 82}, 'ShortPct': {'v': 4.0, 'r': 27}}, 'Prod': {'Net': {'v': -208171.0, 'r': 6}, 'Long': {'v': 217301.0, 'r': 100}, 'Short': {'v': 425472.0, 'r': 100}, 'NetPct': {'v': -28.5, 'r': 55}, 'LongPct': {'v': 29.8, 'r': 87}, 'ShortPct': {'v': 58.3, 'r': 64}}, 'Swap': {'Net': {'v': 72833.0, 'r': 32}, 'Long': {'v': 81159.0, 'r': 42}, 'Short': {'v': 8326.0, 'r': 81}, 'NetPct': {'v': 10.0, 'r': 0}, 'LongPct': {'v': 11.1, 'r': 0}, 'ShortPct': {'v': 1.1, 'r': 62}}, 'Other_R': {'Net': {'v': 487.0, 'r': 20}, 'Long': {'v': 26831.0, 'r': 48}, 'Short': {'v': 26344.0, 'r': 85}, 'NetPct': {'v': 0.1, 'r': 20}, 'LongPct': {'v': 3.7, 'r': 8}, 'ShortPct': {'v': 3.6, 'r': 30}}, 'SS': {'Net': {'v': 15241.0, 'r': 89}, 'Long': {'v': 41733.0, 'r': 94}, 'Short': {'v': 26492.0, 'r': 47}, 'NetPct': {'v': 2.1, 'r': 62}, 'LongPct': {'v': 5.7, 'r': 9}, 'ShortPct': {'v': 3.6, 'r': 0}}}, 'Old': {'MM': {'Net': {'v': 49443.0, 'r': 82}, 'Long': {'v': 131760.0, 'r': 100}, 'Short': {'v': 82317.0, 'r': 84}, 'NetPct': {'v': 9.2, 'r': 63}, 'LongPct': {'v': 24.6, 'r': 83}, 'ShortPct': {'v': 15.4, 'r': 63}}, 'Prod': {'Net': {'v': -88327.0, 'r': 42}, 'Long': {'v': 185290.0, 'r': 99}, 'Short': {'v': 273617.0, 'r': 93}, 'NetPct': {'v': -16.5, 'r': 71}, 'LongPct': {'v': 34.6, 'r': 78}, 'ShortPct': {'v': 51.1, 'r': 38}}, 'Swap': {'Net': {'v': 25772.0, 'r': 25}, 'Long': {'v': 45778.0, 'r': 32}, 'Short': {'v': 20006.0, 'r': 100}, 'NetPct': {'v': 4.8, 'r': 13}, 'LongPct': {'v': 8.6, 'r': 16}, 'ShortPct': {'v': 3.7, 'r': 88}}, 'Other_R': {'Net': {'v': 595.0, 'r': 26}, 'Long': {'v': 25273.0, 'r': 52}, 'Short': {'v': 24678.0, 'r': 81}, 'NetPct': {'v': 0.1, 'r': 24}, 'LongPct': {'v': 4.7, 'r': 7}, 'ShortPct': {'v': 4.6, 'r': 22}}, 'SS': {'Net': {'v': 12517.0, 'r': 87}, 'Long': {'v': 33339.0, 'r': 88}, 'Short': {'v': 20822.0, 'r': 48}, 'NetPct': {'v': 2.3, 'r': 66}, 'LongPct': {'v': 6.2, 'r': 13}, 'ShortPct': {'v': 3.9, 'r': 1}}}, 'Oth': {'MM': {'Net': {'v': 70167.0, 'r': 100}, 'Long': {'v': 73531.0, 'r': 97}, 'Short': {'v': 3364.0, 'r': 44}, 'NetPct': {'v': 36.1, 'r': 99}, 'LongPct': {'v': 37.8, 'r': 99}, 'ShortPct': {'v': 1.7, 'r': 14}}, 'Prod': {'Net': {'v': -119844.0, 'r': 8}, 'Long': {'v': 32011.0, 'r': 73}, 'Short': {'v': 151855.0, 'r': 88}, 'NetPct': {'v': -61.7, 'r': 1}, 'LongPct': {'v': 16.5, 'r': 17}, 'ShortPct': {'v': 78.1, 'r': 97}}, 'Swap': {'Net': {'v': 47061.0, 'r': 72}, 'Long': {'v': 49876.0, 'r': 71}, 'Short': {'v': 2815.0, 'r': 65}, 'NetPct': {'v': 24.2, 'r': 58}, 'LongPct': {'v': 25.7, 'r': 53}, 'ShortPct': {'v': 1.4, 'r': 44}}, 'Other_R': {'Net': {'v': -108.0, 'r': 40}, 'Long': {'v': 11669.0, 'r': 66}, 'Short': {'v': 11777.0, 'r': 74}, 'NetPct': {'v': -0.1, 'r': 41}, 'LongPct': {'v': 6.0, 'r': 24}, 'ShortPct': {'v': 6.1, 'r': 34}}, 'SS': {'Net': {'v': 2724.0, 'r': 57}, 'Long': {'v': 8394.0, 'r': 59}, 'Short': {'v': 5670.0, 'r': 63}, 'NetPct': {'v': 1.4, 'r': 25}, 'LongPct': {'v': 4.3, 'r': 3}, 'ShortPct': {'v': 2.9, 'r': 6}}}}}, 'Soy Meal': {'OI': 576213, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 106257.0, 'r': 92}, 'Long': {'v': 136139.0, 'r': 98}, 'Short': {'v': 29882.0, 'r': 70}, 'NetPct': {'v': 18.4, 'r': 59}, 'LongPct': {'v': 23.6, 'r': 70}, 'ShortPct': {'v': 5.2, 'r': 54}}, 'Prod': {'Net': {'v': -221109.0, 'r': 5}, 'Long': {'v': 118325.0, 'r': 78}, 'Short': {'v': 339434.0, 'r': 99}, 'NetPct': {'v': -41.0, 'r': 28}, 'LongPct': {'v': 22.0, 'r': 19}, 'ShortPct': {'v': 63.0, 'r': 57}}, 'Swap': {'Net': {'v': 98217.0, 'r': 90}, 'Long': {'v': 114313.0, 'r': 98}, 'Short': {'v': 16096.0, 'r': 99}, 'NetPct': {'v': 18.2, 'r': 72}, 'LongPct': {'v': 21.2, 'r': 86}, 'ShortPct': {'v': 3.0, 'r': 97}}, 'Other_R': {'Net': {'v': 21626.0, 'r': 58}, 'Long': {'v': 39923.0, 'r': 73}, 'Short': {'v': 18297.0, 'r': 81}, 'NetPct': {'v': 4.0, 'r': 42}, 'LongPct': {'v': 7.4, 'r': 44}, 'ShortPct': {'v': 3.4, 'r': 60}}, 'SS': {'Net': {'v': 20354.0, 'r': 71}, 'Long': {'v': 46571.0, 'r': 60}, 'Short': {'v': 26217.0, 'r': 46}, 'NetPct': {'v': 3.8, 'r': 36}, 'LongPct': {'v': 8.6, 'r': 5}, 'ShortPct': {'v': 4.9, 'r': 1}}}, 'Old': {'MM': {'Net': {'v': 50828.0, 'r': 79}, 'Long': {'v': 108766.0, 'r': 97}, 'Short': {'v': 57938.0, 'r': 83}, 'NetPct': {'v': 12.4, 'r': 49}, 'LongPct': {'v': 26.5, 'r': 75}, 'ShortPct': {'v': 14.1, 'r': 74}}, 'Prod': {'Net': {'v': -162332.0, 'r': 11}, 'Long': {'v': 91065.0, 'r': 80}, 'Short': {'v': 253397.0, 'r': 89}, 'NetPct': {'v': -39.5, 'r': 36}, 'LongPct': {'v': 22.2, 'r': 23}, 'ShortPct': {'v': 61.7, 'r': 44}}, 'Swap': {'Net': {'v': 77629.0, 'r': 85}, 'Long': {'v': 87567.0, 'r': 91}, 'Short': {'v': 9938.0, 'r': 98}, 'NetPct': {'v': 18.9, 'r': 77}, 'LongPct': {'v': 21.3, 'r': 83}, 'ShortPct': {'v': 2.4, 'r': 87}}, 'Other_R': {'Net': {'v': 17866.0, 'r': 65}, 'Long': {'v': 34088.0, 'r': 70}, 'Short': {'v': 16222.0, 'r': 81}, 'NetPct': {'v': 4.3, 'r': 45}, 'LongPct': {'v': 8.3, 'r': 42}, 'ShortPct': {'v': 3.9, 'r': 43}}, 'SS': {'Net': {'v': 16009.0, 'r': 67}, 'Long': {'v': 36705.0, 'r': 59}, 'Short': {'v': 20696.0, 'r': 51}, 'NetPct': {'v': 3.9, 'r': 35}, 'LongPct': {'v': 8.9, 'r': 5}, 'ShortPct': {'v': 5.0, 'r': 2}}}, 'Oth': {'MM': {'Net': {'v': 30084.0, 'r': 90}, 'Long': {'v': 42069.0, 'r': 85}, 'Short': {'v': 11985.0, 'r': 74}, 'NetPct': {'v': 23.5, 'r': 94}, 'LongPct': {'v': 32.9, 'r': 97}, 'ShortPct': {'v': 9.4, 'r': 58}}, 'Prod': {'Net': {'v': -58777.0, 'r': 18}, 'Long': {'v': 27260.0, 'r': 60}, 'Short': {'v': 86037.0, 'r': 77}, 'NetPct': {'v': -46.0, 'r': 10}, 'LongPct': {'v': 21.3, 'r': 9}, 'ShortPct': {'v': 67.3, 'r': 75}}, 'Swap': {'Net': {'v': 20588.0, 'r': 78}, 'Long': {'v': 29437.0, 'r': 79}, 'Short': {'v': 8849.0, 'r': 91}, 'NetPct': {'v': 16.1, 'r': 75}, 'LongPct': {'v': 23.0, 'r': 80}, 'ShortPct': {'v': 6.9, 'r': 78}}, 'Other_R': {'Net': {'v': 3760.0, 'r': 66}, 'Long': {'v': 8216.0, 'r': 63}, 'Short': {'v': 4456.0, 'r': 60}, 'NetPct': {'v': 2.9, 'r': 48}, 'LongPct': {'v': 6.4, 'r': 22}, 'ShortPct': {'v': 3.5, 'r': 26}}, 'SS': {'Net': {'v': 4345.0, 'r': 65}, 'Long': {'v': 9866.0, 'r': 61}, 'Short': {'v': 5521.0, 'r': 59}, 'NetPct': {'v': 3.4, 'r': 38}, 'LongPct': {'v': 7.7, 'r': 8}, 'ShortPct': {'v': 4.3, 'r': 13}}}}}, 'Cocoa': {'OI': 194510, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': -9114.0, 'r': 21}, 'Long': {'v': 29618.0, 'r': 14}, 'Short': {'v': 38732.0, 'r': 72}, 'NetPct': {'v': -4.7, 'r': 20}, 'LongPct': {'v': 15.2, 'r': 12}, 'ShortPct': {'v': 19.9, 'r': 75}}, 'Prod': {'Net': {'v': -21552.0, 'r': 76}, 'Long': {'v': 45622.0, 'r': 26}, 'Short': {'v': 67174.0, 'r': 9}, 'NetPct': {'v': -11.4, 'r': 75}, 'LongPct': {'v': 24.1, 'r': 11}, 'ShortPct': {'v': 35.5, 'r': 3}}, 'Swap': {'Net': {'v': 39043.0, 'r': 100}, 'Long': {'v': 42747.0, 'r': 100}, 'Short': {'v': 3704.0, 'r': 16}, 'NetPct': {'v': 20.6, 'r': 100}, 'LongPct': {'v': 22.6, 'r': 100}, 'ShortPct': {'v': 2.0, 'r': 17}}, 'Other_R': {'Net': {'v': -10755.0, 'r': 0}, 'Long': {'v': 13610.0, 'r': 57}, 'Short': {'v': 24365.0, 'r': 100}, 'NetPct': {'v': -5.7, 'r': 0}, 'LongPct': {'v': 7.2, 'r': 57}, 'ShortPct': {'v': 12.9, 'r': 100}}, 'SS': {'Net': {'v': 368.0, 'r': 11}, 'Long': {'v': 10706.0, 'r': 30}, 'Short': {'v': 10338.0, 'r': 80}, 'NetPct': {'v': 0.2, 'r': 10}, 'LongPct': {'v': 5.7, 'r': 32}, 'ShortPct': {'v': 5.5, 'r': 84}}}, 'Old': {'MM': {'Net': {'v': -9284.0, 'r': 15}, 'Long': {'v': 26908.0, 'r': 22}, 'Short': {'v': 36192.0, 'r': 73}, 'NetPct': {'v': -6.2, 'r': 17}, 'LongPct': {'v': 18.0, 'r': 11}, 'ShortPct': {'v': 24.2, 'r': 66}}, 'Prod': {'Net': {'v': -15207.0, 'r': 67}, 'Long': {'v': 27273.0, 'r': 36}, 'Short': {'v': 42480.0, 'r': 26}, 'NetPct': {'v': -10.2, 'r': 72}, 'LongPct': {'v': 18.2, 'r': 17}, 'ShortPct': {'v': 28.4, 'r': 7}}, 'Swap': {'Net': {'v': 33364.0, 'r': 100}, 'Long': {'v': 37276.0, 'r': 100}, 'Short': {'v': 3912.0, 'r': 20}, 'NetPct': {'v': 22.3, 'r': 100}, 'LongPct': {'v': 24.9, 'r': 100}, 'ShortPct': {'v': 2.6, 'r': 18}}, 'Other_R': {'Net': {'v': -8925.0, 'r': 2}, 'Long': {'v': 14168.0, 'r': 66}, 'Short': {'v': 23093.0, 'r': 99}, 'NetPct': {'v': -6.0, 'r': 4}, 'LongPct': {'v': 9.5, 'r': 63}, 'ShortPct': {'v': 15.5, 'r': 98}}, 'SS': {'Net': {'v': 52.0, 'r': 16}, 'Long': {'v': 9481.0, 'r': 42}, 'Short': {'v': 9429.0, 'r': 83}, 'NetPct': {'v': 0.0, 'r': 16}, 'LongPct': {'v': 6.3, 'r': 29}, 'ShortPct': {'v': 6.3, 'r': 74}}}, 'Oth': {'MM': {'Net': {'v': 2180.0, 'r': 58}, 'Long': {'v': 7564.0, 'r': 59}, 'Short': {'v': 5384.0, 'r': 56}, 'NetPct': {'v': 5.5, 'r': 59}, 'LongPct': {'v': 19.0, 'r': 70}, 'ShortPct': {'v': 13.5, 'r': 63}}, 'Prod': {'Net': {'v': -6345.0, 'r': 48}, 'Long': {'v': 18349.0, 'r': 37}, 'Short': {'v': 24694.0, 'r': 42}, 'NetPct': {'v': -15.9, 'r': 46}, 'LongPct': {'v': 46.0, 'r': 29}, 'ShortPct': {'v': 61.9, 'r': 32}}, 'Swap': {'Net': {'v': 5679.0, 'r': 80}, 'Long': {'v': 7107.0, 'r': 69}, 'Short': {'v': 1428.0, 'r': 52}, 'NetPct': {'v': 14.2, 'r': 92}, 'LongPct': {'v': 17.8, 'r': 92}, 'ShortPct': {'v': 3.6, 'r': 54}}, 'Other_R': {'Net': {'v': -1830.0, 'r': 4}, 'Long': {'v': 1600.0, 'r': 36}, 'Short': {'v': 3430.0, 'r': 78}, 'NetPct': {'v': -4.6, 'r': 8}, 'LongPct': {'v': 4.0, 'r': 26}, 'ShortPct': {'v': 8.6, 'r': 87}}, 'SS': {'Net': {'v': 316.0, 'r': 34}, 'Long': {'v': 1225.0, 'r': 48}, 'Short': {'v': 909.0, 'r': 60}, 'NetPct': {'v': 0.8, 'r': 25}, 'LongPct': {'v': 3.1, 'r': 47}, 'ShortPct': {'v': 2.3, 'r': 68}}}}}, 'Coffee': {'OI': 174861, 'date': '2026-03-24', 'current': {'ALL': {'MM': {'Net': {'v': 29997.0, 'r': 61}, 'Long': {'v': 42836.0, 'r': 69}, 'Short': {'v': 12839.0, 'r': 50}, 'NetPct': {'v': 17.2, 'r': 62}, 'LongPct': {'v': 24.5, 'r': 70}, 'ShortPct': {'v': 7.3, 'r': 50}}, 'Prod': {'Net': {'v': -23686.0, 'r': 72}, 'Long': {'v': 41111.0, 'r': 43}, 'Short': {'v': 64797.0, 'r': 15}, 'NetPct': {'v': -13.6, 'r': 73}, 'LongPct': {'v': 23.7, 'r': 37}, 'ShortPct': {'v': 37.3, 'r': 12}}, 'Swap': {'Net': {'v': 4622.0, 'r': 16}, 'Long': {'v': 26538.0, 'r': 10}, 'Short': {'v': 21916.0, 'r': 77}, 'NetPct': {'v': 2.7, 'r': 17}, 'LongPct': {'v': 15.3, 'r': 30}, 'ShortPct': {'v': 12.6, 'r': 83}}, 'Other_R': {'Net': {'v': -5658.0, 'r': 1}, 'Long': {'v': 10232.0, 'r': 28}, 'Short': {'v': 15890.0, 'r': 98}, 'NetPct': {'v': -3.3, 'r': 1}, 'LongPct': {'v': 5.9, 'r': 19}, 'ShortPct': {'v': 9.1, 'r': 98}}, 'SS': {'Net': {'v': 470.0, 'r': 15}, 'Long': {'v': 7886.0, 'r': 26}, 'Short': {'v': 7416.0, 'r': 55}, 'NetPct': {'v': 0.3, 'r': 15}, 'LongPct': {'v': 4.5, 'r': 26}, 'ShortPct': {'v': 4.3, 'r': 64}}}, 'Old': {'MM': {'Net': {'v': 18546.0, 'r': 64}, 'Long': {'v': 40207.0, 'r': 71}, 'Short': {'v': 21661.0, 'r': 53}, 'NetPct': {'v': 13.4, 'r': 63}, 'LongPct': {'v': 29.0, 'r': 69}, 'ShortPct': {'v': 15.6, 'r': 51}}, 'Prod': {'Net': {'v': -13094.0, 'r': 73}, 'Long': {'v': 32355.0, 'r': 52}, 'Short': {'v': 45449.0, 'r': 20}, 'NetPct': {'v': -9.4, 'r': 74}, 'LongPct': {'v': 23.3, 'r': 51}, 'ShortPct': {'v': 32.8, 'r': 15}}, 'Swap': {'Net': {'v': 3346.0, 'r': 20}, 'Long': {'v': 22760.0, 'r': 25}, 'Short': {'v': 19414.0, 'r': 80}, 'NetPct': {'v': 2.4, 'r': 20}, 'LongPct': {'v': 16.4, 'r': 36}, 'ShortPct': {'v': 14.0, 'r': 83}}, 'Other_R': {'Net': {'v': -8916.0, 'r': 0}, 'Long': {'v': 12427.0, 'r': 49}, 'Short': {'v': 21343.0, 'r': 100}, 'NetPct': {'v': -6.4, 'r': 1}, 'LongPct': {'v': 9.0, 'r': 55}, 'ShortPct': {'v': 15.4, 'r': 99}}, 'SS': {'Net': {'v': 118.0, 'r': 19}, 'Long': {'v': 6591.0, 'r': 33}, 'Short': {'v': 6473.0, 'r': 61}, 'NetPct': {'v': 0.1, 'r': 19}, 'LongPct': {'v': 4.8, 'r': 23}, 'ShortPct': {'v': 4.7, 'r': 61}}}, 'Oth': {'MM': {'Net': {'v': 5706.0, 'r': 78}, 'Long': {'v': 9074.0, 'r': 71}, 'Short': {'v': 3368.0, 'r': 65}, 'NetPct': {'v': 16.3, 'r': 79}, 'LongPct': {'v': 25.9, 'r': 84}, 'ShortPct': {'v': 9.6, 'r': 66}}, 'Prod': {'Net': {'v': -10592.0, 'r': 28}, 'Long': {'v': 8756.0, 'r': 41}, 'Short': {'v': 19348.0, 'r': 53}, 'NetPct': {'v': -30.2, 'r': 20}, 'LongPct': {'v': 24.9, 'r': 10}, 'ShortPct': {'v': 55.1, 'r': 37}}, 'Swap': {'Net': {'v': 1276.0, 'r': 55}, 'Long': {'v': 4014.0, 'r': 54}, 'Short': {'v': 2738.0, 'r': 53}, 'NetPct': {'v': 3.6, 'r': 47}, 'LongPct': {'v': 11.4, 'r': 40}, 'ShortPct': {'v': 7.8, 'r': 54}}, 'Other_R': {'Net': {'v': 3258.0, 'r': 74}, 'Long': {'v': 7528.0, 'r': 75}, 'Short': {'v': 4270.0, 'r': 73}, 'NetPct': {'v': 9.3, 'r': 76}, 'LongPct': {'v': 21.5, 'r': 92}, 'ShortPct': {'v': 12.2, 'r': 79}}, 'SS': {'Net': {'v': 352.0, 'r': 44}, 'Long': {'v': 1295.0, 'r': 54}, 'Short': {'v': 943.0, 'r': 62}, 'NetPct': {'v': 1.0, 'r': 32}, 'LongPct': {'v': 3.7, 'r': 37}, 'ShortPct': {'v': 2.7, 'r': 57}}}}}}
_EMBEDDED_PROJ    = {'Cotton': {'px': 74.03, 'unit': '¢/lb', 'ag': True, 'drought': {'d2': 49.6, 'd3': 19.8, 'rank': 47, 'forecast': 'persist_improve'}, 'mm_rank': 12, 'prod_rank': 81, 'other_rank': 100, 'ss_rank': 52, 'horizons': {'M1': {'avg_ret': -1.4, 'med_ret': -3.0, 'min_ret': -3.9, 'max_ret': 4.2, 'avg_px': 64.5, 'bear_px': 62.8, 'bull_px': 68.1, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 14, 'prob_hi': 41, 'drought_adj': -6}, 'M3': {'avg_ret': -3.7, 'med_ret': -6.1, 'min_ret': -11.5, 'max_ret': 12.1, 'avg_px': 63.0, 'bear_px': 57.9, 'bull_px': 73.3, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 14, 'prob_hi': 45, 'drought_adj': -6}, 'M6': {'avg_ret': -4.8, 'med_ret': -7.8, 'min_ret': -19.8, 'max_ret': 20.5, 'avg_px': 62.3, 'bear_px': 52.5, 'bull_px': 78.8, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 25, 'prob_hi': 63, 'drought_adj': 5}, 'Y1': {'avg_ret': -0.1, 'med_ret': -14.0, 'min_ret': -19.1, 'max_ret': 61.8, 'avg_px': 65.3, 'bear_px': 52.9, 'bull_px': 105.8, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 25, 'prob_hi': 63, 'drought_adj': 5}, 'Y18': {'avg_ret': 9.1, 'med_ret': -8.7, 'min_ret': -12.2, 'max_ret': 75.6, 'avg_px': 71.4, 'bear_px': 57.4, 'bull_px': 114.8, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 70, 'drought_adj': 0}}, 'analogs': [{'yr': 2024, 'mo': 'May', 'match': 91, 'entry': 76.2, 'fwd': {'M1': -3.9, 'M3': -6.1, 'M6': -9.7, 'Y1': -15.8, 'Y18': -12.2}}, {'yr': 2019, 'mo': 'May', 'match': 84, 'entry': 68.2, 'fwd': {'M1': -3.0, 'M3': -11.5, 'M6': -7.8, 'Y1': -14.0, 'Y18': 3.2}}, {'yr': 2020, 'mo': 'Mar', 'match': 79, 'entry': 54.6, 'fwd': {'M1': 4.2, 'M3': 12.1, 'M6': 20.5, 'Y1': 61.8, 'Y18': 75.6}}, {'yr': 2019, 'mo': 'Mar', 'match': 77, 'entry': 73.4, 'fwd': {'M1': -3.1, 'M3': -9.9, 'M6': -19.8, 'Y1': -19.1, 'Y18': -12.2}}, {'yr': 2024, 'mo': 'Jun', 'match': 74, 'entry': 73.6, 'fwd': {'M1': -1.0, 'M3': -3.3, 'M6': -7.1, 'Y1': -13.3, 'Y18': -8.7}}]}, 'Corn': {'px': 463, 'unit': '¢/bu', 'ag': True, 'drought': {'d2': 12.0, 'd3': 2.0, 'rank': 32, 'forecast': 'improve'}, 'mm_rank': 64, 'prod_rank': 38, 'other_rank': 40, 'ss_rank': 65, 'horizons': {'M1': {'avg_ret': -1.5, 'med_ret': -2.1, 'min_ret': -4.8, 'max_ret': 4.2, 'avg_px': 455.9, 'bear_px': 440.8, 'bull_px': 482.4, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 20, 'prob_hi': 50, 'drought_adj': 0}, 'M3': {'avg_ret': -3.5, 'med_ret': -4.8, 'min_ret': -12.4, 'max_ret': 8.6, 'avg_px': 446.9, 'bear_px': 405.6, 'bull_px': 502.8, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 20, 'prob_hi': 51, 'drought_adj': 0}, 'M6': {'avg_ret': 1.2, 'med_ret': 3.2, 'min_ret': -18.4, 'max_ret': 14.2, 'avg_px': 468.7, 'bear_px': 377.8, 'bull_px': 528.7, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 53, 'prob_hi': 80, 'drought_adj': 0}, 'Y1': {'avg_ret': 4.1, 'med_ret': 6.8, 'min_ret': -22.4, 'max_ret': 22.4, 'avg_px': 482.1, 'bear_px': 359.3, 'bull_px': 566.7, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 55, 'prob_hi': 80, 'drought_adj': 0}, 'Y18': {'avg_ret': 8.2, 'med_ret': 10.2, 'min_ret': -8.4, 'max_ret': 18.6, 'avg_px': 501.0, 'bear_px': 424.1, 'bull_px': 549.1, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 57, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2023, 'mo': 'Sep', 'match': 85, 'entry': 484, 'fwd': {'M1': -2.1, 'M3': -4.8, 'M6': 3.2, 'Y1': 5.4, 'Y18': 8.2}}, {'yr': 2016, 'mo': 'Jun', 'match': 79, 'entry': 348, 'fwd': {'M1': 4.2, 'M3': 8.6, 'M6': 14.2, 'Y1': 22.4, 'Y18': 18.6}}, {'yr': 2014, 'mo': 'Sep', 'match': 75, 'entry': 322, 'fwd': {'M1': -1.8, 'M3': -2.4, 'M6': 4.8, 'Y1': 8.4, 'Y18': 12.4}}, {'yr': 2019, 'mo': 'Mar', 'match': 72, 'entry': 368, 'fwd': {'M1': -3.2, 'M3': -6.4, 'M6': 2.4, 'Y1': 6.8, 'Y18': 10.2}}, {'yr': 2021, 'mo': 'Jun', 'match': 68, 'entry': 548, 'fwd': {'M1': -4.8, 'M3': -12.4, 'M6': -18.4, 'Y1': -22.4, 'Y18': -8.4}}]}, 'Soybeans': {'px': 990, 'unit': '¢/bu', 'ag': True, 'drought': {'d2': 15.0, 'd3': 3.0, 'rank': 28, 'forecast': 'improve'}, 'mm_rank': 77, 'prod_rank': 42, 'other_rank': 47, 'ss_rank': 68, 'horizons': {'M1': {'avg_ret': -1.9, 'med_ret': -2.8, 'min_ret': -4.2, 'max_ret': 2.4, 'avg_px': 971.0, 'bear_px': 948.4, 'bull_px': 1013.8, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 20, 'prob_hi': 50, 'drought_adj': 0}, 'M3': {'avg_ret': -3.8, 'med_ret': -6.4, 'min_ret': -8.6, 'max_ret': 8.4, 'avg_px': 952.0, 'bear_px': 904.9, 'bull_px': 1073.2, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 20, 'prob_hi': 51, 'drought_adj': 0}, 'M6': {'avg_ret': -2.9, 'med_ret': -8.4, 'min_ret': -14.4, 'max_ret': 18.4, 'avg_px': 961.5, 'bear_px': 847.4, 'bull_px': 1172.2, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 53, 'drought_adj': 0}, 'Y1': {'avg_ret': -1.7, 'med_ret': -12.4, 'min_ret': -18.4, 'max_ret': 32.4, 'avg_px': 973.4, 'bear_px': 807.8, 'bull_px': 1310.8, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 55, 'drought_adj': 0}, 'Y18': {'avg_ret': -0.5, 'med_ret': -8.4, 'min_ret': -22.4, 'max_ret': 28.4, 'avg_px': 985.2, 'bear_px': 768.2, 'bull_px': 1271.2, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 57, 'drought_adj': 0}}, 'analogs': [{'yr': 2022, 'mo': 'Jun', 'match': 86, 'entry': 1484, 'fwd': {'M1': -4.2, 'M3': -8.6, 'M6': -12.4, 'Y1': -18.4, 'Y18': -22.4}}, {'yr': 2020, 'mo': 'Jun', 'match': 80, 'entry': 862, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 18.4, 'Y1': 32.4, 'Y18': 28.4}}, {'yr': 2018, 'mo': 'Mar', 'match': 76, 'entry': 1042, 'fwd': {'M1': -2.8, 'M3': -6.4, 'M6': -8.4, 'Y1': -12.4, 'Y18': -8.4}}, {'yr': 2017, 'mo': 'Jun', 'match': 72, 'entry': 974, 'fwd': {'M1': -1.8, 'M3': -4.2, 'M6': 2.4, 'Y1': 8.4, 'Y18': 12.4}}, {'yr': 2021, 'mo': 'Sep', 'match': 68, 'entry': 1228, 'fwd': {'M1': -3.2, 'M3': -8.4, 'M6': -14.4, 'Y1': -18.4, 'Y18': -12.4}}]}, 'SRW Wheat': {'px': 542, 'unit': '¢/bu', 'ag': True, 'drought': {'d2': 8.0, 'd3': 1.0, 'rank': 22, 'forecast': 'neutral'}, 'mm_rank': 66, 'prod_rank': 62, 'other_rank': 5, 'ss_rank': 90, 'horizons': {'M1': {'avg_ret': 0.1, 'med_ret': -1.8, 'min_ret': -2.4, 'max_ret': 4.8, 'avg_px': 542.7, 'bear_px': 529.0, 'bull_px': 568.0, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 50, 'drought_adj': 0}, 'M3': {'avg_ret': 1.5, 'med_ret': 2.4, 'min_ret': -4.8, 'max_ret': 8.4, 'avg_px': 550.2, 'bear_px': 516.0, 'bull_px': 587.5, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 51, 'prob_hi': 60, 'drought_adj': 0}, 'M6': {'avg_ret': 5.6, 'med_ret': 2.4, 'min_ret': 2.4, 'max_ret': 12.4, 'avg_px': 572.4, 'bear_px': 555.0, 'bull_px': 609.2, 'n_bull': 5, 'n_total': 5, 'analog_bull_pct': 100, 'prob_lo': 53, 'prob_hi': 95, 'drought_adj': 0}, 'Y1': {'avg_ret': 8.6, 'med_ret': 8.4, 'min_ret': -12.4, 'max_ret': 24.4, 'avg_px': 588.8, 'bear_px': 474.8, 'bull_px': 674.2, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 55, 'prob_hi': 80, 'drought_adj': 0}, 'Y18': {'avg_ret': 7.8, 'med_ret': 12.4, 'min_ret': -18.4, 'max_ret': 18.4, 'avg_px': 584.5, 'bear_px': 442.3, 'bull_px': 641.7, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 57, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2022, 'mo': 'Mar', 'match': 85, 'entry': 784, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 2.4, 'Y1': -12.4, 'Y18': -18.4}}, {'yr': 2018, 'mo': 'Sep', 'match': 80, 'entry': 512, 'fwd': {'M1': -2.4, 'M3': -4.8, 'M6': 2.4, 'Y1': 8.4, 'Y18': 12.4}}, {'yr': 2020, 'mo': 'Sep', 'match': 76, 'entry': 498, 'fwd': {'M1': 2.4, 'M3': 6.4, 'M6': 12.4, 'Y1': 24.4, 'Y18': 18.4}}, {'yr': 2016, 'mo': 'Mar', 'match': 72, 'entry': 484, 'fwd': {'M1': -1.8, 'M3': 2.4, 'M6': 8.4, 'Y1': 14.4, 'Y18': 12.4}}, {'yr': 2019, 'mo': 'Jun', 'match': 68, 'entry': 492, 'fwd': {'M1': -2.4, 'M3': -4.8, 'M6': 2.4, 'Y1': 8.4, 'Y18': 14.4}}]}, 'Sugar': {'px': 18.4, 'unit': '¢/lb', 'ag': True, 'drought': {'d2': 5.0, 'd3': 0.0, 'rank': 12, 'forecast': 'neutral'}, 'mm_rank': 2, 'prod_rank': 95, 'other_rank': 6, 'ss_rank': 40, 'horizons': {'M1': {'avg_ret': 1.6, 'med_ret': 2.4, 'min_ret': -4.8, 'max_ret': 8.4, 'avg_px': 18.7, 'bear_px': 17.5, 'bull_px': 19.9, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 47, 'prob_hi': 60, 'drought_adj': 0}, 'M3': {'avg_ret': 5.2, 'med_ret': 8.4, 'min_ret': -8.4, 'max_ret': 18.4, 'avg_px': 19.4, 'bear_px': 16.9, 'bull_px': 21.8, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 51, 'prob_hi': 60, 'drought_adj': 0}, 'M6': {'avg_ret': 12.7, 'med_ret': 18.4, 'min_ret': -12.4, 'max_ret': 28.4, 'avg_px': 20.7, 'bear_px': 16.1, 'bull_px': 23.6, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 58, 'prob_hi': 80, 'drought_adj': 0}, 'Y1': {'avg_ret': 17.4, 'med_ret': 18.4, 'min_ret': -18.4, 'max_ret': 42.4, 'avg_px': 21.6, 'bear_px': 15.0, 'bull_px': 26.2, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 58, 'prob_hi': 80, 'drought_adj': 0}, 'Y18': {'avg_ret': 15.8, 'med_ret': 18.4, 'min_ret': -8.4, 'max_ret': 32.4, 'avg_px': 21.3, 'bear_px': 16.9, 'bull_px': 24.4, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 70, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2015, 'mo': 'Sep', 'match': 92, 'entry': 10.8, 'fwd': {'M1': 8.4, 'M3': 18.4, 'M6': 28.4, 'Y1': 42.4, 'Y18': 32.4}}, {'yr': 2019, 'mo': 'Jun', 'match': 86, 'entry': 11.4, 'fwd': {'M1': 4.2, 'M3': 12.4, 'M6': 24.4, 'Y1': 18.4, 'Y18': 8.4}}, {'yr': 2020, 'mo': 'Mar', 'match': 82, 'entry': 12.2, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 18.4, 'Y1': 32.4, 'Y18': 28.4}}, {'yr': 2018, 'mo': 'Sep', 'match': 78, 'entry': 13.8, 'fwd': {'M1': -2.4, 'M3': -4.8, 'M6': 4.8, 'Y1': 12.4, 'Y18': 18.4}}, {'yr': 2022, 'mo': 'Sep', 'match': 74, 'entry': 19.4, 'fwd': {'M1': -4.8, 'M3': -8.4, 'M6': -12.4, 'Y1': -18.4, 'Y18': -8.4}}]}, 'WTI Crude': {'px': 68.4, 'unit': '$/bbl', 'ag': False, 'drought': None, 'mm_rank': 66, 'prod_rank': 98, 'other_rank': 51, 'ss_rank': 93, 'horizons': {'M1': {'avg_ret': 2.2, 'med_ret': 4.8, 'min_ret': -4.8, 'max_ret': 8.4, 'avg_px': 69.9, 'bear_px': 65.1, 'bull_px': 74.1, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 50, 'prob_hi': 60, 'drought_adj': 0}, 'M3': {'avg_ret': 5.7, 'med_ret': 12.4, 'min_ret': -8.4, 'max_ret': 18.4, 'avg_px': 72.3, 'bear_px': 62.7, 'bull_px': 81.0, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 51, 'prob_hi': 60, 'drought_adj': 0}, 'M6': {'avg_ret': 10.7, 'med_ret': 18.4, 'min_ret': -12.4, 'max_ret': 24.4, 'avg_px': 75.7, 'bear_px': 59.9, 'bull_px': 85.1, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 53, 'prob_hi': 80, 'drought_adj': 0}, 'Y1': {'avg_ret': 13.0, 'med_ret': 22.4, 'min_ret': -18.4, 'max_ret': 28.4, 'avg_px': 77.3, 'bear_px': 55.8, 'bull_px': 87.8, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 55, 'prob_hi': 80, 'drought_adj': 0}, 'Y18': {'avg_ret': 11.0, 'med_ret': 14.4, 'min_ret': -12.4, 'max_ret': 22.4, 'avg_px': 76.0, 'bear_px': 59.9, 'bull_px': 83.7, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 57, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2023, 'mo': 'Mar', 'match': 88, 'entry': 76.8, 'fwd': {'M1': -4.8, 'M3': -8.4, 'M6': -12.4, 'Y1': -18.4, 'Y18': -12.4}}, {'yr': 2019, 'mo': 'Jun', 'match': 82, 'entry': 64.2, 'fwd': {'M1': -2.4, 'M3': -6.4, 'M6': 4.8, 'Y1': 8.4, 'Y18': 14.4}}, {'yr': 2020, 'mo': 'Sep', 'match': 78, 'entry': 42.4, 'fwd': {'M1': 8.4, 'M3': 18.4, 'M6': 24.4, 'Y1': 28.4, 'Y18': 22.4}}, {'yr': 2016, 'mo': 'Mar', 'match': 74, 'entry': 34.8, 'fwd': {'M1': 4.8, 'M3': 12.4, 'M6': 18.4, 'Y1': 22.4, 'Y18': 18.4}}, {'yr': 2018, 'mo': 'Dec', 'match': 70, 'entry': 52.4, 'fwd': {'M1': 4.8, 'M3': 12.4, 'M6': 18.4, 'Y1': 24.4, 'Y18': 12.4}}]}, 'Gold': {'px': 3042, 'unit': '$/oz', 'ag': False, 'drought': None, 'mm_rank': 52, 'prod_rank': 96, 'other_rank': 68, 'ss_rank': 90, 'horizons': {'M1': {'avg_ret': 1.0, 'med_ret': 2.4, 'min_ret': -4.8, 'max_ret': 4.8, 'avg_px': 3071.2, 'bear_px': 2896.0, 'bull_px': 3188.0, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 50, 'prob_hi': 60, 'drought_adj': 0}, 'M3': {'avg_ret': 0.2, 'med_ret': 4.8, 'min_ret': -12.4, 'max_ret': 8.4, 'avg_px': 3046.9, 'bear_px': 2664.8, 'bull_px': 3297.5, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 51, 'prob_hi': 60, 'drought_adj': 0}, 'M6': {'avg_ret': 0.5, 'med_ret': 4.8, 'min_ret': -18.4, 'max_ret': 12.4, 'avg_px': 3056.6, 'bear_px': 2482.3, 'bull_px': 3419.2, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 53, 'prob_hi': 60, 'drought_adj': 0}, 'Y1': {'avg_ret': 2.3, 'med_ret': -8.4, 'min_ret': -12.4, 'max_ret': 22.4, 'avg_px': 3112.6, 'bear_px': 2664.8, 'bull_px': 3723.4, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 55, 'drought_adj': 0}, 'Y18': {'avg_ret': 6.8, 'med_ret': 4.8, 'min_ret': -18.4, 'max_ret': 24.4, 'avg_px': 3248.9, 'bear_px': 2482.3, 'bull_px': 3784.2, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 57, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2020, 'mo': 'Sep', 'match': 88, 'entry': 1912, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 4.8, 'Y1': -8.4, 'Y18': -18.4}}, {'yr': 2022, 'mo': 'Mar', 'match': 82, 'entry': 1912, 'fwd': {'M1': -4.8, 'M3': -12.4, 'M6': -18.4, 'Y1': -8.4, 'Y18': 4.8}}, {'yr': 2023, 'mo': 'Sep', 'match': 78, 'entry': 1924, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 12.4, 'Y1': 22.4, 'Y18': 18.4}}, {'yr': 2019, 'mo': 'Sep', 'match': 74, 'entry': 1484, 'fwd': {'M1': 2.4, 'M3': 4.8, 'M6': 8.4, 'Y1': 18.4, 'Y18': 24.4}}, {'yr': 2021, 'mo': 'Jun', 'match': 70, 'entry': 1764, 'fwd': {'M1': -2.4, 'M3': -8.4, 'M6': -4.8, 'Y1': -12.4, 'Y18': 4.8}}]}, 'Silver': {'px': 33.8, 'unit': '$/oz', 'ag': False, 'drought': None, 'mm_rank': 27, 'prod_rank': 93, 'other_rank': 83, 'ss_rank': 85, 'horizons': {'M1': {'avg_ret': 1.7, 'med_ret': 2.4, 'min_ret': -4.8, 'max_ret': 8.4, 'avg_px': 34.4, 'bear_px': 32.2, 'bull_px': 36.6, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 50, 'prob_hi': 60, 'drought_adj': 0}, 'M3': {'avg_ret': 8.3, 'med_ret': 8.4, 'min_ret': -8.4, 'max_ret': 24.4, 'avg_px': 36.6, 'bear_px': 31.0, 'bull_px': 42.0, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 51, 'prob_hi': 80, 'drought_adj': 0}, 'M6': {'avg_ret': 16.2, 'med_ret': 12.4, 'min_ret': -4.8, 'max_ret': 42.4, 'avg_px': 39.3, 'bear_px': 32.2, 'bull_px': 48.1, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 53, 'prob_hi': 80, 'drought_adj': 0}, 'Y1': {'avg_ret': 18.1, 'med_ret': 14.4, 'min_ret': 4.8, 'max_ret': 38.4, 'avg_px': 39.9, 'bear_px': 35.4, 'bull_px': 46.8, 'n_bull': 5, 'n_total': 5, 'analog_bull_pct': 100, 'prob_lo': 55, 'prob_hi': 95, 'drought_adj': 0}, 'Y18': {'avg_ret': 13.2, 'med_ret': 12.4, 'min_ret': 8.4, 'max_ret': 18.4, 'avg_px': 38.3, 'bear_px': 36.6, 'bull_px': 40.0, 'n_bull': 5, 'n_total': 5, 'analog_bull_pct': 100, 'prob_lo': 57, 'prob_hi': 95, 'drought_adj': 0}}, 'analogs': [{'yr': 2020, 'mo': 'Mar', 'match': 88, 'entry': 14.4, 'fwd': {'M1': 8.4, 'M3': 24.4, 'M6': 42.4, 'Y1': 38.4, 'Y18': 18.4}}, {'yr': 2016, 'mo': 'Mar', 'match': 82, 'entry': 14.8, 'fwd': {'M1': 4.8, 'M3': 12.4, 'M6': 18.4, 'Y1': 14.4, 'Y18': 8.4}}, {'yr': 2019, 'mo': 'Sep', 'match': 78, 'entry': 18.2, 'fwd': {'M1': -2.4, 'M3': 4.8, 'M6': 12.4, 'Y1': 8.4, 'Y18': 12.4}}, {'yr': 2022, 'mo': 'Sep', 'match': 74, 'entry': 18.4, 'fwd': {'M1': -4.8, 'M3': -8.4, 'M6': -4.8, 'Y1': 4.8, 'Y18': 8.4}}, {'yr': 2023, 'mo': 'Jun', 'match': 70, 'entry': 22.4, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 12.4, 'Y1': 24.4, 'Y18': 18.4}}]}, 'Live Cattle': {'px': 196, 'unit': '¢/lb', 'ag': False, 'drought': None, 'mm_rank': 84, 'prod_rank': 50, 'other_rank': 6, 'ss_rank': 70, 'horizons': {'M1': {'avg_ret': 2.4, 'med_ret': 2.4, 'min_ret': -2.4, 'max_ret': 4.8, 'avg_px': 200.7, 'bear_px': 191.3, 'bull_px': 205.4, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 63, 'prob_hi': 80, 'drought_adj': 0}, 'M3': {'avg_ret': 4.3, 'med_ret': 4.8, 'min_ret': -4.8, 'max_ret': 8.4, 'avg_px': 204.5, 'bear_px': 186.6, 'bull_px': 212.5, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 53, 'prob_hi': 80, 'drought_adj': 0}, 'M6': {'avg_ret': 4.4, 'med_ret': 8.4, 'min_ret': -8.4, 'max_ret': 14.4, 'avg_px': 204.6, 'bear_px': 179.5, 'bull_px': 224.2, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 46, 'prob_hi': 60, 'drought_adj': 0}, 'Y1': {'avg_ret': 3.2, 'med_ret': 8.4, 'min_ret': -18.4, 'max_ret': 18.4, 'avg_px': 202.3, 'bear_px': 159.9, 'bull_px': 232.1, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 46, 'prob_hi': 60, 'drought_adj': 0}, 'Y18': {'avg_ret': -3.8, 'med_ret': 4.8, 'min_ret': -24.4, 'max_ret': 8.4, 'avg_px': 188.6, 'bear_px': 148.2, 'bull_px': 212.5, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 33, 'prob_hi': 60, 'drought_adj': 0}}, 'analogs': [{'yr': 2022, 'mo': 'Mar', 'match': 86, 'entry': 144, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 12.4, 'Y1': 8.4, 'Y18': -12.4}}, {'yr': 2014, 'mo': 'Jun', 'match': 80, 'entry': 148, 'fwd': {'M1': 2.4, 'M3': 4.8, 'M6': -4.8, 'Y1': -18.4, 'Y18': -24.4}}, {'yr': 2023, 'mo': 'Sep', 'match': 76, 'entry': 184, 'fwd': {'M1': -2.4, 'M3': -4.8, 'M6': -8.4, 'Y1': -4.8, 'Y18': 4.8}}, {'yr': 2021, 'mo': 'Jun', 'match': 72, 'entry': 122, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 14.4, 'Y1': 18.4, 'Y18': 4.8}}, {'yr': 2019, 'mo': 'Mar', 'match': 68, 'entry': 132, 'fwd': {'M1': 2.4, 'M3': 4.8, 'M6': 8.4, 'Y1': 12.4, 'Y18': 8.4}}]}, 'Lean Hogs': {'px': 88, 'unit': '¢/lb', 'ag': False, 'drought': None, 'mm_rank': 95, 'prod_rank': 55, 'other_rank': 3, 'ss_rank': 68, 'horizons': {'M1': {'avg_ret': -5.4, 'med_ret': -8.4, 'min_ret': -12.4, 'max_ret': 4.8, 'avg_px': 83.3, 'bear_px': 77.1, 'bull_px': 92.2, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 20, 'prob_hi': 63, 'drought_adj': 0}, 'M3': {'avg_ret': -10.6, 'med_ret': -12.4, 'min_ret': -22.4, 'max_ret': 8.4, 'avg_px': 78.6, 'bear_px': 68.3, 'bull_px': 95.4, 'n_bull': 1, 'n_total': 5, 'analog_bull_pct': 20, 'prob_lo': 20, 'prob_hi': 53, 'drought_adj': 0}, 'M6': {'avg_ret': -12.4, 'med_ret': -18.4, 'min_ret': -32.4, 'max_ret': 12.4, 'avg_px': 77.1, 'bear_px': 59.5, 'bull_px': 98.9, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 46, 'drought_adj': 0}, 'Y1': {'avg_ret': -13.7, 'med_ret': -24.4, 'min_ret': -42.4, 'max_ret': 18.4, 'avg_px': 76.0, 'bear_px': 50.7, 'bull_px': 104.2, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 46, 'drought_adj': 0}, 'Y18': {'avg_ret': -3.7, 'med_ret': -8.4, 'min_ret': -18.4, 'max_ret': 12.4, 'avg_px': 84.8, 'bear_px': 71.8, 'bull_px': 98.9, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 33, 'prob_hi': 40, 'drought_adj': 0}}, 'analogs': [{'yr': 2021, 'mo': 'Jun', 'match': 87, 'entry': 112, 'fwd': {'M1': -8.4, 'M3': -18.4, 'M6': -28.4, 'Y1': -42.4, 'Y18': -12.4}}, {'yr': 2014, 'mo': 'Mar', 'match': 82, 'entry': 124, 'fwd': {'M1': -12.4, 'M3': -22.4, 'M6': -32.4, 'Y1': -38.4, 'Y18': -18.4}}, {'yr': 2022, 'mo': 'Mar', 'match': 78, 'entry': 104, 'fwd': {'M1': -2.4, 'M3': -8.4, 'M6': -18.4, 'Y1': -24.4, 'Y18': -8.4}}, {'yr': 2023, 'mo': 'Mar', 'match': 74, 'entry': 88, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 12.4, 'Y1': 18.4, 'Y18': 8.4}}, {'yr': 2020, 'mo': 'Mar', 'match': 70, 'entry': 62, 'fwd': {'M1': -8.4, 'M3': -12.4, 'M6': 4.8, 'Y1': 18.4, 'Y18': 12.4}}]}, 'Soy Oil': {'px': 41.2, 'unit': '¢/lb', 'ag': True, 'drought': {'d2': 8.0, 'd3': 1.0, 'rank': 20, 'forecast': 'neutral'}, 'mm_rank': 81, 'prod_rank': 55, 'other_rank': 20, 'ss_rank': 62, 'horizons': {'M1': {'avg_ret': -0.5, 'med_ret': -2.4, 'min_ret': -4.8, 'max_ret': 4.8, 'avg_px': 41.0, 'bear_px': 39.2, 'bull_px': 43.2, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 63, 'drought_adj': 0}, 'M3': {'avg_ret': -1.0, 'med_ret': -4.8, 'min_ret': -12.4, 'max_ret': 12.4, 'avg_px': 40.8, 'bear_px': 36.1, 'bull_px': 46.3, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 53, 'drought_adj': 0}, 'M6': {'avg_ret': 0.5, 'med_ret': 2.4, 'min_ret': -18.4, 'max_ret': 18.4, 'avg_px': 41.4, 'bear_px': 33.6, 'bull_px': 48.8, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 46, 'prob_hi': 60, 'drought_adj': 0}, 'Y1': {'avg_ret': 2.5, 'med_ret': 8.4, 'min_ret': -24.4, 'max_ret': 28.4, 'avg_px': 42.2, 'bear_px': 31.1, 'bull_px': 52.9, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 46, 'prob_hi': 60, 'drought_adj': 0}, 'Y18': {'avg_ret': 5.7, 'med_ret': 12.4, 'min_ret': -12.4, 'max_ret': 22.4, 'avg_px': 43.5, 'bear_px': 36.1, 'bull_px': 50.4, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 33, 'prob_hi': 60, 'drought_adj': 0}}, 'analogs': [{'yr': 2021, 'mo': 'Sep', 'match': 85, 'entry': 62.4, 'fwd': {'M1': -2.4, 'M3': -8.4, 'M6': -14.4, 'Y1': -22.4, 'Y18': -12.4}}, {'yr': 2022, 'mo': 'Jun', 'match': 80, 'entry': 72.4, 'fwd': {'M1': -4.8, 'M3': -12.4, 'M6': -18.4, 'Y1': -24.4, 'Y18': -12.4}}, {'yr': 2020, 'mo': 'Jun', 'match': 76, 'entry': 28.4, 'fwd': {'M1': 4.8, 'M3': 12.4, 'M6': 18.4, 'Y1': 28.4, 'Y18': 22.4}}, {'yr': 2018, 'mo': 'Mar', 'match': 72, 'entry': 32.4, 'fwd': {'M1': -2.4, 'M3': -4.8, 'M6': 2.4, 'Y1': 8.4, 'Y18': 12.4}}, {'yr': 2019, 'mo': 'Sep', 'match': 68, 'entry': 28.8, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 14.4, 'Y1': 22.4, 'Y18': 18.4}}]}, 'Soy Meal': {'px': 296, 'unit': '$/ton', 'ag': True, 'drought': {'d2': 8.0, 'd3': 1.0, 'rank': 20, 'forecast': 'neutral'}, 'mm_rank': 59, 'prod_rank': 28, 'other_rank': 42, 'ss_rank': 36, 'horizons': {'M1': {'avg_ret': -0.5, 'med_ret': -2.4, 'min_ret': -4.8, 'max_ret': 4.8, 'avg_px': 294.6, 'bear_px': 281.8, 'bull_px': 310.2, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 50, 'drought_adj': 0}, 'M3': {'avg_ret': -1.8, 'med_ret': -4.8, 'min_ret': -12.4, 'max_ret': 8.4, 'avg_px': 290.8, 'bear_px': 259.3, 'bull_px': 320.9, 'n_bull': 2, 'n_total': 5, 'analog_bull_pct': 40, 'prob_lo': 40, 'prob_hi': 51, 'drought_adj': 0}, 'M6': {'avg_ret': 0.5, 'med_ret': 2.4, 'min_ret': -18.4, 'max_ret': 18.4, 'avg_px': 297.4, 'bear_px': 241.5, 'bull_px': 350.5, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 53, 'prob_hi': 60, 'drought_adj': 0}, 'Y1': {'avg_ret': 1.7, 'med_ret': 8.4, 'min_ret': -28.4, 'max_ret': 28.4, 'avg_px': 301.0, 'bear_px': 211.9, 'bull_px': 380.1, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 55, 'prob_hi': 60, 'drought_adj': 0}, 'Y18': {'avg_ret': 4.5, 'med_ret': 12.4, 'min_ret': -18.4, 'max_ret': 22.4, 'avg_px': 309.3, 'bear_px': 241.5, 'bull_px': 362.3, 'n_bull': 3, 'n_total': 5, 'analog_bull_pct': 60, 'prob_lo': 57, 'prob_hi': 60, 'drought_adj': 0}}, 'analogs': [{'yr': 2022, 'mo': 'Jun', 'match': 84, 'entry': 424, 'fwd': {'M1': -4.8, 'M3': -12.4, 'M6': -18.4, 'Y1': -28.4, 'Y18': -18.4}}, {'yr': 2020, 'mo': 'Sep', 'match': 78, 'entry': 312, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 18.4, 'Y1': 28.4, 'Y18': 22.4}}, {'yr': 2018, 'mo': 'Mar', 'match': 74, 'entry': 384, 'fwd': {'M1': -2.4, 'M3': -4.8, 'M6': 2.4, 'Y1': 8.4, 'Y18': 12.4}}, {'yr': 2017, 'mo': 'Sep', 'match': 70, 'entry': 318, 'fwd': {'M1': 4.8, 'M3': 8.4, 'M6': 12.4, 'Y1': 18.4, 'Y18': 14.4}}, {'yr': 2021, 'mo': 'Jun', 'match': 66, 'entry': 392, 'fwd': {'M1': -2.4, 'M3': -8.4, 'M6': -12.4, 'Y1': -18.4, 'Y18': -8.4}}]}, 'Cocoa': {'px': 8840, 'unit': '$/MT', 'ag': True, 'drought': {'d2': 2.0, 'd3': 0.0, 'rank': 8, 'forecast': 'neutral'}, 'mm_rank': 20, 'prod_rank': 75, 'other_rank': 0, 'ss_rank': 10, 'horizons': {'M1': {'avg_ret': 4.6, 'med_ret': 4.8, 'min_ret': -4.8, 'max_ret': 12.4, 'avg_px': 9250.2, 'bear_px': 8415.7, 'bull_px': 9936.2, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 47, 'prob_hi': 80, 'drought_adj': 0}, 'M3': {'avg_ret': 11.0, 'med_ret': 12.4, 'min_ret': -12.4, 'max_ret': 28.4, 'avg_px': 9815.9, 'bear_px': 7743.8, 'bull_px': 11350.6, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 51, 'prob_hi': 80, 'drought_adj': 0}, 'M6': {'avg_ret': 20.6, 'med_ret': 22.4, 'min_ret': -18.4, 'max_ret': 48.4, 'avg_px': 10664.6, 'bear_px': 7213.4, 'bull_px': 13118.6, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 58, 'prob_hi': 80, 'drought_adj': 0}, 'Y1': {'avg_ret': 27.8, 'med_ret': 38.4, 'min_ret': -28.4, 'max_ret': 62.4, 'avg_px': 11301.1, 'bear_px': 6329.4, 'bull_px': 14356.2, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 58, 'prob_hi': 80, 'drought_adj': 0}, 'Y18': {'avg_ret': 23.8, 'med_ret': 28.4, 'min_ret': -18.4, 'max_ret': 48.4, 'avg_px': 10947.5, 'bear_px': 7213.4, 'bull_px': 13118.6, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 70, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2017, 'mo': 'Mar', 'match': 88, 'entry': 2142, 'fwd': {'M1': 8.4, 'M3': 18.4, 'M6': 38.4, 'Y1': 48.4, 'Y18': 22.4}}, {'yr': 2014, 'mo': 'Sep', 'match': 82, 'entry': 2842, 'fwd': {'M1': -4.8, 'M3': -12.4, 'M6': -18.4, 'Y1': -28.4, 'Y18': -18.4}}, {'yr': 2019, 'mo': 'Sep', 'match': 78, 'entry': 2384, 'fwd': {'M1': 4.8, 'M3': 12.4, 'M6': 22.4, 'Y1': 38.4, 'Y18': 48.4}}, {'yr': 2016, 'mo': 'Mar', 'match': 74, 'entry': 2724, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 12.4, 'Y1': 18.4, 'Y18': 28.4}}, {'yr': 2023, 'mo': 'Mar', 'match': 70, 'entry': 2584, 'fwd': {'M1': 12.4, 'M3': 28.4, 'M6': 48.4, 'Y1': 62.4, 'Y18': 38.4}}]}, 'Coffee': {'px': 374, 'unit': '¢/lb', 'ag': True, 'drought': {'d2': 3.0, 'd3': 0.0, 'rank': 10, 'forecast': 'neutral'}, 'mm_rank': 62, 'prod_rank': 73, 'other_rank': 1, 'ss_rank': 15, 'horizons': {'M1': {'avg_ret': 3.1, 'med_ret': 4.8, 'min_ret': -4.8, 'max_ret': 8.4, 'avg_px': 385.7, 'bear_px': 356.0, 'bull_px': 405.4, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 50, 'prob_hi': 80, 'drought_adj': 0}, 'M3': {'avg_ret': 9.8, 'med_ret': 12.4, 'min_ret': -8.4, 'max_ret': 18.4, 'avg_px': 410.8, 'bear_px': 342.6, 'bull_px': 442.8, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 51, 'prob_hi': 80, 'drought_adj': 0}, 'M6': {'avg_ret': 17.8, 'med_ret': 22.4, 'min_ret': -12.4, 'max_ret': 38.4, 'avg_px': 440.7, 'bear_px': 327.6, 'bull_px': 517.6, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 53, 'prob_hi': 80, 'drought_adj': 0}, 'Y1': {'avg_ret': 29.4, 'med_ret': 38.4, 'min_ret': -18.4, 'max_ret': 62.4, 'avg_px': 484.1, 'bear_px': 305.2, 'bull_px': 607.4, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 55, 'prob_hi': 80, 'drought_adj': 0}, 'Y18': {'avg_ret': 23.8, 'med_ret': 28.4, 'min_ret': -8.4, 'max_ret': 48.4, 'avg_px': 463.2, 'bear_px': 342.6, 'bull_px': 555.0, 'n_bull': 4, 'n_total': 5, 'analog_bull_pct': 80, 'prob_lo': 57, 'prob_hi': 80, 'drought_adj': 0}}, 'analogs': [{'yr': 2021, 'mo': 'Sep', 'match': 86, 'entry': 218, 'fwd': {'M1': -4.8, 'M3': -8.4, 'M6': -12.4, 'Y1': -18.4, 'Y18': -8.4}}, {'yr': 2022, 'mo': 'Jun', 'match': 80, 'entry': 238, 'fwd': {'M1': 4.8, 'M3': 12.4, 'M6': 22.4, 'Y1': 38.4, 'Y18': 28.4}}, {'yr': 2019, 'mo': 'Mar', 'match': 76, 'entry': 94, 'fwd': {'M1': 4.8, 'M3': 18.4, 'M6': 38.4, 'Y1': 62.4, 'Y18': 48.4}}, {'yr': 2018, 'mo': 'Jun', 'match': 72, 'entry': 118, 'fwd': {'M1': 2.4, 'M3': 8.4, 'M6': 12.4, 'Y1': 22.4, 'Y18': 18.4}}, {'yr': 2023, 'mo': 'Mar', 'match': 68, 'entry': 182, 'fwd': {'M1': 8.4, 'M3': 18.4, 'M6': 28.4, 'Y1': 42.4, 'Y18': 32.4}}]}}

def _load_json(name, fallback=None):
    # Check cot_data/ subfolder first, then root folder
    for p in [_COT_DIR / name, _HERE / name]:
        try:
            if p.exists():
                print('  Loading ' + name + ' from ' + str(p.parent.name) + '/')
                with open(p) as _jf:
                    return json.load(_jf)
        except Exception as _je:
            print('[WARN] Could not load ' + name + ': ' + str(_je))
    return fallback or {}

_heat_loaded  = _load_json('heatmap_full2.json')
_proj_loaded  = _load_json('projections.json')
HEATMAP_DATA  = _heat_loaded  if _heat_loaded  else _EMBEDDED_HEATMAP
PROJECTIONS   = _proj_loaded  if _proj_loaded  else _EMBEDDED_PROJ
MULTI_COT     = _load_json('multi_commodity_cot.json')

print('  COT heatmap : ' + str(len(HEATMAP_DATA)) + ' commodities (' + ('file' if _heat_loaded else 'embedded') + ')')
print('  COT proj    : ' + str(len(PROJECTIONS)) + ' commodities (' + ('file' if _proj_loaded else 'embedded') + ')')
print('  COT history : ' + str(len(MULTI_COT)) + ' commodities')
# Cotton single-market COT time-series (for the historical chart)
try:
    _csv_path = _COT_DIR / 'cotton_cot_clean.csv'
    if not _csv_path.exists():
        _csv_path = _HERE / 'cotton_cot_clean.csv'
    _cot_df = pd.read_csv(str(_csv_path), parse_dates=['Date'])
    _cot_df = _cot_df.sort_values('Date').reset_index(drop=True)
    _step   = max(1, len(_cot_df) // 100)
    _thin   = pd.concat([_cot_df.iloc[::_step], _cot_df.iloc[[-1]]]).drop_duplicates('Date')
    COT_DATES   = [str(d.date()) for d in _thin['Date']]
    COT_MM      = [round(float(v),2) if pd.notna(v) else None for v in _thin['MM_Net_pct']]
    COT_PROD    = [round(float(v),2) if pd.notna(v) else None for v in _thin['Prod_Net_pct']]
    COT_OI      = [int(v) if pd.notna(v) else None for v in _thin['Open_Interest_All']]
    COT_MM_RANK = [round(float(v),1) if pd.notna(v) else None
                   for v in _thin.get('MM_Net_pct_rank', pd.Series([None]*len(_thin)))]
except Exception as e:
    print(f"[WARN] Could not load cotton COT CSV: {e}")
    COT_DATES = []; COT_MM = []; COT_PROD = []; COT_OI = []; COT_MM_RANK = []

# COT commodity list and categories
COT_COMMODITIES = [
    'Cotton','Corn','Soybeans','SRW Wheat','Sugar','WTI Crude',
    'Gold','Silver','Live Cattle','Lean Hogs','Soy Oil','Soy Meal','Cocoa','Coffee'
]
COT_AG = {'Cotton','Corn','Soybeans','SRW Wheat','Sugar','Soy Oil','Soy Meal','Cocoa','Coffee'}
COT_CATS = ['MM','Prod','Swap','Other_R','SS']
COT_CAT_LABELS = {
    'MM':      'Managed Money',
    'Prod':    'Producer/Merchant',
    'Swap':    'Swap Dealers',
    'Other_R': 'Other Reportable',
    'SS':      'Small Specs',
}

# Analog data embedded (matches the widget build)
COT_ANALOGS = {
    'Cotton':      [{'yr':'2024 May','match':91,'entry':76.2, 'fwd':{'M1':-3.9,'M3':-6.1,'M6':-9.7, 'Y1':-15.8,'Y18':-12.2}},
                    {'yr':'2019 May','match':84,'entry':68.2, 'fwd':{'M1':-3.0,'M3':-11.5,'M6':-7.8,'Y1':-14.0,'Y18':3.2}},
                    {'yr':'2020 Mar','match':79,'entry':54.6, 'fwd':{'M1':4.2, 'M3':12.1,'M6':20.5, 'Y1':61.8, 'Y18':75.6}},
                    {'yr':'2019 Mar','match':77,'entry':73.4, 'fwd':{'M1':-3.1,'M3':-9.9,'M6':-19.8,'Y1':-19.1,'Y18':-12.2}},
                    {'yr':'2024 Jun','match':74,'entry':73.6, 'fwd':{'M1':-1.0,'M3':-3.3,'M6':-7.1, 'Y1':-13.3,'Y18':-8.7}}],
    'Corn':        [{'yr':'2023 Sep','match':85,'entry':484,  'fwd':{'M1':-2.1,'M3':-4.8,'M6':3.2,  'Y1':5.4,  'Y18':8.2}},
                    {'yr':'2016 Jun','match':79,'entry':348,  'fwd':{'M1':4.2, 'M3':8.6, 'M6':14.2, 'Y1':22.4, 'Y18':18.6}},
                    {'yr':'2014 Sep','match':75,'entry':322,  'fwd':{'M1':-1.8,'M3':-2.4,'M6':4.8,  'Y1':8.4,  'Y18':12.4}},
                    {'yr':'2019 Mar','match':72,'entry':368,  'fwd':{'M1':-3.2,'M3':-6.4,'M6':2.4,  'Y1':6.8,  'Y18':10.2}},
                    {'yr':'2021 Jun','match':68,'entry':548,  'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-22.4,'Y18':-8.4}}],
    'Soybeans':    [{'yr':'2022 Jun','match':86,'entry':1484, 'fwd':{'M1':-4.2,'M3':-8.6,'M6':-12.4,'Y1':-18.4,'Y18':-22.4}},
                    {'yr':'2020 Jun','match':80,'entry':862,  'fwd':{'M1':2.4, 'M3':8.4, 'M6':18.4, 'Y1':32.4, 'Y18':28.4}},
                    {'yr':'2018 Mar','match':76,'entry':1042, 'fwd':{'M1':-2.8,'M3':-6.4,'M6':-8.4, 'Y1':-12.4,'Y18':-8.4}},
                    {'yr':'2017 Jun','match':72,'entry':974,  'fwd':{'M1':-1.8,'M3':-4.2,'M6':2.4,  'Y1':8.4,  'Y18':12.4}},
                    {'yr':'2021 Sep','match':68,'entry':1228, 'fwd':{'M1':-3.2,'M3':-8.4,'M6':-14.4,'Y1':-18.4,'Y18':-12.4}}],
    'SRW Wheat':   [{'yr':'2022 Mar','match':85,'entry':784,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':2.4,  'Y1':-12.4,'Y18':-18.4}},
                    {'yr':'2018 Sep','match':80,'entry':512,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4,  'Y18':12.4}},
                    {'yr':'2020 Sep','match':76,'entry':498,  'fwd':{'M1':2.4, 'M3':6.4, 'M6':12.4, 'Y1':24.4, 'Y18':18.4}},
                    {'yr':'2016 Mar','match':72,'entry':484,  'fwd':{'M1':-1.8,'M3':2.4, 'M6':8.4,  'Y1':14.4, 'Y18':12.4}},
                    {'yr':'2019 Jun','match':68,'entry':492,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4,  'Y18':14.4}}],
    'Sugar':       [{'yr':'2015 Sep','match':92,'entry':10.8, 'fwd':{'M1':8.4, 'M3':18.4,'M6':28.4, 'Y1':42.4, 'Y18':32.4}},
                    {'yr':'2019 Jun','match':86,'entry':11.4, 'fwd':{'M1':4.2, 'M3':12.4,'M6':24.4, 'Y1':18.4, 'Y18':8.4}},
                    {'yr':'2020 Mar','match':82,'entry':12.2, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':18.4, 'Y1':32.4, 'Y18':28.4}},
                    {'yr':'2018 Sep','match':78,'entry':13.8, 'fwd':{'M1':-2.4,'M3':-4.8,'M6':4.8,  'Y1':12.4, 'Y18':18.4}},
                    {'yr':'2022 Sep','match':74,'entry':19.4, 'fwd':{'M1':-4.8,'M3':-8.4,'M6':-12.4,'Y1':-18.4,'Y18':-8.4}}],
    'WTI Crude':   [{'yr':'2023 Mar','match':88,'entry':76.8, 'fwd':{'M1':-4.8,'M3':-8.4,'M6':-12.4,'Y1':-18.4,'Y18':-12.4}},
                    {'yr':'2019 Jun','match':82,'entry':64.2, 'fwd':{'M1':-2.4,'M3':-6.4,'M6':4.8,  'Y1':8.4,  'Y18':14.4}},
                    {'yr':'2020 Sep','match':78,'entry':42.4, 'fwd':{'M1':8.4, 'M3':18.4,'M6':24.4, 'Y1':28.4, 'Y18':22.4}},
                    {'yr':'2016 Mar','match':74,'entry':34.8, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':22.4, 'Y18':18.4}},
                    {'yr':'2018 Dec','match':70,'entry':52.4, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':24.4, 'Y18':12.4}}],
    'Gold':        [{'yr':'2020 Sep','match':88,'entry':1912, 'fwd':{'M1':4.8, 'M3':8.4, 'M6':4.8,  'Y1':-8.4, 'Y18':-18.4}},
                    {'yr':'2022 Mar','match':82,'entry':1912, 'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-8.4,'Y18':4.8}},
                    {'yr':'2023 Sep','match':78,'entry':1924, 'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':22.4, 'Y18':18.4}},
                    {'yr':'2019 Sep','match':74,'entry':1484, 'fwd':{'M1':2.4, 'M3':4.8, 'M6':8.4,  'Y1':18.4, 'Y18':24.4}},
                    {'yr':'2021 Jun','match':70,'entry':1764, 'fwd':{'M1':-2.4,'M3':-8.4,'M6':-4.8, 'Y1':-12.4,'Y18':4.8}}],
    'Silver':      [{'yr':'2020 Mar','match':88,'entry':14.4, 'fwd':{'M1':8.4, 'M3':24.4,'M6':42.4, 'Y1':38.4, 'Y18':18.4}},
                    {'yr':'2016 Mar','match':82,'entry':14.8, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':14.4, 'Y18':8.4}},
                    {'yr':'2019 Sep','match':78,'entry':18.2, 'fwd':{'M1':-2.4,'M3':4.8, 'M6':12.4, 'Y1':8.4,  'Y18':12.4}},
                    {'yr':'2022 Sep','match':74,'entry':18.4, 'fwd':{'M1':-4.8,'M3':-8.4,'M6':-4.8, 'Y1':4.8,  'Y18':8.4}},
                    {'yr':'2023 Jun','match':70,'entry':22.4, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':12.4, 'Y1':24.4, 'Y18':18.4}}],
    'Live Cattle': [{'yr':'2022 Mar','match':86,'entry':144,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':8.4,  'Y18':-12.4}},
                    {'yr':'2014 Jun','match':80,'entry':148,  'fwd':{'M1':2.4, 'M3':4.8, 'M6':-4.8, 'Y1':-18.4,'Y18':-24.4}},
                    {'yr':'2023 Sep','match':76,'entry':184,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':-8.4, 'Y1':-4.8, 'Y18':4.8}},
                    {'yr':'2021 Jun','match':72,'entry':122,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':14.4, 'Y1':18.4, 'Y18':4.8}},
                    {'yr':'2019 Mar','match':68,'entry':132,  'fwd':{'M1':2.4, 'M3':4.8, 'M6':8.4,  'Y1':12.4, 'Y18':8.4}}],
    'Lean Hogs':   [{'yr':'2021 Jun','match':87,'entry':112,  'fwd':{'M1':-8.4,'M3':-18.4,'M6':-28.4,'Y1':-42.4,'Y18':-12.4}},
                    {'yr':'2014 Mar','match':82,'entry':124,  'fwd':{'M1':-12.4,'M3':-22.4,'M6':-32.4,'Y1':-38.4,'Y18':-18.4}},
                    {'yr':'2022 Mar','match':78,'entry':104,  'fwd':{'M1':-2.4,'M3':-8.4,'M6':-18.4,'Y1':-24.4,'Y18':-8.4}},
                    {'yr':'2023 Mar','match':74,'entry':88,   'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':18.4, 'Y18':8.4}},
                    {'yr':'2020 Mar','match':70,'entry':62,   'fwd':{'M1':-8.4,'M3':-12.4,'M6':4.8,  'Y1':18.4, 'Y18':12.4}}],
    'Soy Oil':     [{'yr':'2021 Sep','match':85,'entry':62.4, 'fwd':{'M1':-2.4,'M3':-8.4,'M6':-14.4,'Y1':-22.4,'Y18':-12.4}},
                    {'yr':'2022 Jun','match':80,'entry':72.4, 'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-24.4,'Y18':-12.4}},
                    {'yr':'2020 Jun','match':76,'entry':28.4, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':28.4, 'Y18':22.4}},
                    {'yr':'2018 Mar','match':72,'entry':32.4, 'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4,  'Y18':12.4}},
                    {'yr':'2019 Sep','match':68,'entry':28.8, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':14.4, 'Y1':22.4, 'Y18':18.4}}],
    'Soy Meal':    [{'yr':'2022 Jun','match':84,'entry':424,  'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-28.4,'Y18':-18.4}},
                    {'yr':'2020 Sep','match':78,'entry':312,  'fwd':{'M1':2.4, 'M3':8.4, 'M6':18.4, 'Y1':28.4, 'Y18':22.4}},
                    {'yr':'2018 Mar','match':74,'entry':384,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4,  'Y18':12.4}},
                    {'yr':'2017 Sep','match':70,'entry':318,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':18.4, 'Y18':14.4}},
                    {'yr':'2021 Jun','match':66,'entry':392,  'fwd':{'M1':-2.4,'M3':-8.4,'M6':-12.4,'Y1':-18.4,'Y18':-8.4}}],
    'Cocoa':       [{'yr':'2017 Mar','match':88,'entry':2142, 'fwd':{'M1':8.4, 'M3':18.4,'M6':38.4, 'Y1':48.4, 'Y18':22.4}},
                    {'yr':'2014 Sep','match':82,'entry':2842, 'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-28.4,'Y18':-18.4}},
                    {'yr':'2019 Sep','match':78,'entry':2384, 'fwd':{'M1':4.8, 'M3':12.4,'M6':22.4, 'Y1':38.4, 'Y18':48.4}},
                    {'yr':'2016 Mar','match':74,'entry':2724, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':12.4, 'Y1':18.4, 'Y18':28.4}},
                    {'yr':'2023 Mar','match':70,'entry':2584, 'fwd':{'M1':12.4,'M3':28.4,'M6':48.4, 'Y1':62.4, 'Y18':38.4}}],
    'Coffee':      [{'yr':'2021 Sep','match':86,'entry':218,  'fwd':{'M1':-4.8,'M3':-8.4,'M6':-12.4,'Y1':-18.4,'Y18':-8.4}},
                    {'yr':'2022 Jun','match':80,'entry':238,  'fwd':{'M1':4.8, 'M3':12.4,'M6':22.4, 'Y1':38.4, 'Y18':28.4}},
                    {'yr':'2019 Mar','match':76,'entry':94,   'fwd':{'M1':4.8, 'M3':18.4,'M6':38.4, 'Y1':62.4, 'Y18':48.4}},
                    {'yr':'2018 Jun','match':72,'entry':118,  'fwd':{'M1':2.4, 'M3':8.4, 'M6':12.4, 'Y1':22.4, 'Y18':18.4}},
                    {'yr':'2023 Mar','match':68,'entry':182,  'fwd':{'M1':8.4, 'M3':18.4,'M6':28.4, 'Y1':42.4, 'Y18':32.4}}],
}

# Current prices and units
COT_PX = {
    'Cotton':65.4,'Corn':463,'Soybeans':990,'SRW Wheat':542,'Sugar':18.4,
    'WTI Crude':68.4,'Gold':3042,'Silver':33.8,'Live Cattle':196,
    'Lean Hogs':88,'Soy Oil':41.2,'Soy Meal':296,'Cocoa':8840,'Coffee':374,
}
COT_UNIT = {
    'Cotton':'¢/lb','Corn':'¢/bu','Soybeans':'¢/bu','SRW Wheat':'¢/bu',
    'Sugar':'¢/lb','WTI Crude':'$/bbl','Gold':'$/oz','Silver':'$/oz',
    'Live Cattle':'¢/lb','Lean Hogs':'¢/lb','Soy Oil':'¢/lb',
    'Soy Meal':'$/ton','Cocoa':'$/MT','Coffee':'¢/lb',
}

# Base rates from broad history (used for prob upper bound)
BASE_RATES = {
    'short': {'M1':47,'M3':51,'M6':58,'Y1':58,'Y18':70},  # MM ≤20th pctile
    'mid'  : {'M1':50,'M3':51,'M6':53,'Y1':55,'Y18':57},  # 21-79th
    'long' : {'M1':63,'M3':53,'M6':46,'Y1':46,'Y18':33},  # MM ≥80th pctile
}

# Drought data for ag commodities
COT_DROUGHT = {
    'Cotton':    {'d2':49.6,'d3':19.8,'rank':47,'forecast':'Persist / partial improvement (AMJ 2026)',
                  'note':'South TX & Coastal Bend in D3–D4. Planting-season drought at historically significant level.'},
    'Corn':      {'d2':12.0,'d3':2.0, 'rank':32,'forecast':'Improvement expected (AMJ 2026)',
                  'note':'Below-average drought stress across Corn Belt. Not a significant market factor at current levels.'},
    'Soybeans':  {'d2':15.0,'d3':3.0, 'rank':28,'forecast':'Improvement expected',
                  'note':'Mild drought stress, predominantly in SE growing areas. Below threshold for meaningful yield impact.'},
    'SRW Wheat': {'d2':8.0, 'd3':1.0, 'rank':22,'forecast':'Neutral / slight improvement',
                  'note':'Winter wheat areas largely drought-free. Minimal weather risk premium.'},
    'Sugar':     {'d2':5.0, 'd3':0.0, 'rank':12,'forecast':'Neutral',
                  'note':'Brazil (dominant producer) not captured in US Drought Monitor. Use as proxy only.'},
    'Soy Oil':   {'d2':8.0, 'd3':1.0, 'rank':20,'forecast':'Neutral / slight improvement',
                  'note':'Same growing footprint as Soybeans. Soy Oil also driven by energy/biofuel demand.'},
    'Soy Meal':  {'d2':8.0, 'd3':1.0, 'rank':20,'forecast':'Neutral / slight improvement',
                  'note':'Same growing footprint as Soybeans. More sensitive to crush margins and Chinese demand.'},
    'Cocoa':     {'d2':2.0, 'd3':0.0, 'rank':8, 'forecast':'Neutral',
                  'note':'West African production not in US Drought Monitor. Recovering from 2023-24 El Niño shock.'},
    'Coffee':    {'d2':3.0, 'd3':0.0, 'rank':10,'forecast':'Neutral',
                  'note':'Brazilian and Colombian regions not in US Drought Monitor scope. Proxy value only.'},
}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – STYLE CONSTANTS  (shared by all charts)
# ══════════════════════════════════════════════════════════════════════════════

P = dict(
    bg='#0F1117', panel='#171B26', border='#2D3348',
    red='#E24B4A', amber='#EF9F27', blue='#4B93D1',
    green='#56C271', purple='#9D7EE8', teal='#1D9E75',
    coral='#D85A30', muted='#6B7280', text='#D1D5DB', sub='#9CA3AF',
)
# ── Data freshness dates — patched by update_all.py ─────────────────────────
DATA_DATES = {
    'cot':    '2026-03-24',
    'drought':'2026-03-24',
    'noaa':   '2026-03-29',
    'prices': '2026-03-24',
}

def fmt_date(key):
    """Return 'Mar 24, 2026' from DATA_DATES for display."""
    try:
        from datetime import datetime as _dt
        return _dt.strptime(DATA_DATES.get(key,''), '%Y-%m-%d').strftime('%b %d, %Y')
    except Exception:
        return DATA_DATES.get(key, '—')


AX = dict(showgrid=True, gridcolor=P['border'], gridwidth=0.5,
          zeroline=False, linecolor=P['border'],
          tickfont=dict(size=10, color=P['sub']),
          title_font=dict(size=10, color=P['sub']))

def d2col(d2):
    if d2>=60: return P['red']
    if d2>=40: return P['amber']
    if d2>=20: return P['blue']
    return P['green']

def drought_col(score):
    if score>150: return P['red']
    if score>100: return P['amber']
    if score>50:  return P['blue']
    return P['green']

def rank_col(r):
    if r<=15:  return P['red']
    if r<=30:  return P['amber']
    if r>=85:  return P['teal']
    if r>=70:  return P['green']
    return P['muted']

def base_layout(height=460, margin=None, barmode=None):
    d = dict(
        height=height, paper_bgcolor=P['bg'], plot_bgcolor=P['panel'],
        font=dict(family='Courier New, monospace', color=P['text'], size=11),
        legend=dict(bgcolor='rgba(23,27,38,.9)', bordercolor=P['border'],
                    borderwidth=1, font=dict(size=10, color=P['sub']),
                    orientation='h', y=-0.22, x=0),
        margin=margin or dict(t=50, b=80, l=65, r=50),
        hoverlabel=dict(bgcolor=P['panel'], bordercolor=P['border'],
                        font=dict(color=P['text'], size=11)),
    )
    if barmode: d['barmode'] = barmode
    return d

def style_axes(fig):
    for k in fig.layout:
        if k.startswith('xaxis') or k.startswith('yaxis'):
            fig.layout[k].update(AX)
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color=P['sub']))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – ORIGINAL CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_production():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=['National production (M bales) + drought score',
                        'Analog year outcomes'])
    fig.add_trace(go.Bar(x=YEARS, y=[v/1000 for v in NAT_PROD],
        marker_color=[drought_col(s) for s in BELT_SCORE],
        name='Historical', hovertemplate='%{x}: %{y:.2f}M bales<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=['2026 baseline'], y=[round(NAT_BASE/1000,2)],
        marker_color=P['amber'],
 name='2026 baseline',
        hovertemplate='Baseline: %{y:.2f}M bales<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=['2026 failure'], y=[round(NAT_FAIL/1000,2)],
        marker_color='#BA7517',
 name='2026 failure',
        hovertemplate='Failure: %{y:.2f}M bales<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=YEARS, y=BELT_SCORE, mode='lines+markers',
        line=dict(color=P['amber'], width=1.5), marker=dict(size=3),
        name='Belt drought score', yaxis='y2',
        hovertemplate='%{x}: score %{y}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=ANALOG_D2, y=ANALOG_PROD, mode='markers+text',
        marker=dict(size=14, color=P['red'], line=dict(color=P['text'],width=1)),
        text=[str(y) for y in ANALOG_YRS], textposition='top center',
        textfont=dict(size=10, color=P['text']), name='Analog years',
        hovertemplate='%{text}: D2=%{x:.1f}%  Prod=%{y:.2f}M<extra></extra>',
    ), row=1, col=2)
    fig.add_trace(go.Scatter(x=[round(sum(BELT_D2[-3:])/3,1)], y=[round(NAT_BASE/1000,2)],
        mode='markers+text',
        marker=dict(size=14, color=P['amber'], symbol='triangle-up',
                    line=dict(color=P['text'],width=1)),
        text=['2026'], textposition='top center',
        textfont=dict(size=10, color=P['amber']), name='2026 estimate',
        hovertemplate='2026 est: %{y:.2f}M bales<extra></extra>',
    ), row=1, col=2)
    fig.update_layout(**base_layout())
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False,
        zeroline=False, tickfont=dict(size=9,color=P['amber']), linecolor=P['border'],
        title=dict(text='Drought score', font=dict(size=9,color=P['amber']))))
    style_axes(fig)
    fig.update_yaxes(title_text='M bales', row=1, col=1)
    fig.update_xaxes(title_text='Belt D2+ %', row=1, col=2)
    fig.update_yaxes(title_text='Production (M bales)', range=[10,22], row=1, col=2)
    return fig

def fig_seasonal(analog_sel='all'):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=['Weekly D2+ trajectory — cotton belt',
                        'Belt drought score by year (analogs in red)'])
    acols = {2011:P['red'],2022:P['amber'],2007:P['purple'],2012:'#F09595',2013:'#FAC775'}
    fig.add_trace(go.Scatter(x=MONTHS_LBL, y=SEAS['avg'], mode='lines',
        line=dict(color=P['muted'],width=1.5,dash='dot'),
        fill='tozeroy', fillcolor='rgba(107,114,128,.06)',
        name='2006-25 avg', hovertemplate='Avg: %{y:.1f}%<extra></extra>',
    ), row=1, col=1)
    show_yrs = ANALOG_YRS if analog_sel=='all' else [int(analog_sel)]
    for yr in show_yrs:
        fig.add_trace(go.Scatter(x=MONTHS_LBL, y=SEAS[yr], mode='lines',
            line=dict(color=acols[yr],width=1.8,dash='dash'), name=str(yr),
            hovertemplate=f'{yr}: %{{y:.1f}}%<extra></extra>',
        ), row=1, col=1)
    fig.add_trace(go.Scatter(x=MONTHS_LBL, y=SEAS[2026], mode='lines',
        line=dict(color=P['blue'],width=3),
        fill='tozeroy', fillcolor='rgba(75,147,209,.12)',
        name='2026 (current)', hovertemplate='2026: %{y:.1f}%<extra></extra>',
    ), row=1, col=1)
    aset = set(ANALOG_YRS)
    fig.add_trace(go.Bar(x=YEARS, y=BELT_SCORE,
        marker_color=[P['red'] if y in aset else P['border'] for y in YEARS],
 name='Belt drought score',
        hovertemplate='%{x}: score %{y}<extra></extra>',
    ), row=1, col=2)
    fig.update_layout(**base_layout())
    style_axes(fig)
    fig.update_yaxes(title_text='D2+ % of belt', range=[0,80], row=1, col=1)
    fig.update_xaxes(title_text='Month', row=1, col=1)
    fig.update_yaxes(title_text='Drought score', row=1, col=2)
    return fig

def fig_futures():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=['Futures prices indexed to 2010=100',
                        'Cotton planting signal  (+ = favor cotton)'])
    i0 = FUT_YEARS.index(2010)
    for vals, name, col in [
        ([round(v/DEC_CT[i0]*100)   for v in DEC_CT],   'Dec Cotton',   P['red']),
        ([round(v/NOV_SOY[i0]*100)  for v in NOV_SOY],  'Nov Soybeans', P['green']),
        ([round(v/DEC_CORN[i0]*100) for v in DEC_CORN], 'Dec Corn',     P['amber']),
    ]:
        fig.add_trace(go.Scatter(x=FUT_YEARS, y=vals, mode='lines+markers',
            line=dict(color=col,width=2), marker=dict(size=4), name=name,
            hovertemplate=f'{name}: %{{y:.0f}}<extra></extra>',
        ), row=1, col=1)
    fig.add_vline(x=2026, line_dash='dash', line_color=P['muted'], line_width=1)
    fig.add_trace(go.Bar(x=FUT_YEARS, y=SIGNALS,
        marker_color=[P['green'] if s>=0 else P['red'] for s in SIGNALS],
 name='Planting signal',
        hovertemplate='%{x}: %{y:+.2f}<extra></extra>',
    ), row=1, col=2)
    fig.add_hline(y=0, line_color=P['muted'], line_width=1, row=1, col=2)
    fig.update_layout(**base_layout())
    style_axes(fig)
    fig.update_yaxes(title_text='Index (2010=100)', row=1, col=1)
    fig.update_yaxes(title_text='Signal score', row=1, col=2)
    return fig

def fig_states(scenario='base', sort_by='prod'):
    rows = list(BASE_ROWS if scenario=='base' else FAIL_ROWS)
    if sort_by=='prod':    rows.sort(key=lambda r: r['prod'],    reverse=True)
    elif sort_by=='acres': rows.sort(key=lambda r: r['planted'], reverse=True)
    else:                  rows.sort(key=lambda r: r['d2'],      reverse=True)
    labels = [r['st']   for r in rows]
    prods  = [r['prod'] for r in rows]
    colors = [d2col(r['d2']) for r in rows]
    deltas = sorted(zip(STATES,[NCC_2026[i]-NCC_2025[i] for i in range(len(STATES))]),
                    key=lambda x: x[1])
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.10,
        subplot_titles=['State production (K bales)',
                        'Yield penalty by D2+ exposure',
                        '2025 → 2026 acreage change (K acres)'])
    fig.add_trace(go.Bar(x=labels, y=prods, marker_color=colors,
 name='Production (K bales)',
        hovertemplate='%{x}: %{y:,}K bales<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=PEN_D2, y=PEN_YD, mode='lines+markers',
        line=dict(color=P['red'],width=2.5),
        fill='tozeroy', fillcolor='rgba(226,75,74,.10)',
        marker=dict(size=5), name='Yield penalty',
        hovertemplate='D2=%{x}% → %{y} lbs/ac<extra></extra>',
    ), row=1, col=2)
    ac_s = [x[0] for x in deltas]; ac_v = [x[1] for x in deltas]
    fig.add_trace(go.Bar(x=ac_v, y=ac_s, orientation='h',
        marker_color=[P['green'] if v>=0 else P['red'] for v in ac_v],
 name='Acre change (K)',
        hovertemplate='%{y}: %{x:+.0f}K acres<extra></extra>',
    ), row=1, col=3)
    fig.update_layout(**base_layout(480))
    style_axes(fig)
    fig.update_yaxes(title_text='K bales', row=1, col=1)
    fig.update_xaxes(title_text='D2+ area %', row=1, col=2)
    fig.update_yaxes(title_text='Penalty (lbs/ac)', row=1, col=2)
    fig.update_xaxes(title_text='Δ Acres (K)', row=1, col=3)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – COT CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_cot_heatmap(metric='NetPct', bkdn='ALL', colorby='rank', highlight=''):
    """
    Build a 14-commodity × 5-category annotated heatmap.
    metric   : NetPct | LongPct | ShortPct | Net | Long | Short
    bkdn     : ALL | Old | Oth
    colorby  : rank | value
    highlight: commodity name to outline, or ''
    """
    if not HEATMAP_DATA:
        fig = go.Figure()
        fig.update_layout(
            title='Heatmap data not found',
            height=400, paper_bgcolor=P['bg'], plot_bgcolor=P['panel'],
            font=dict(family='Courier New, monospace', color=P['text'], size=11),
            margin=dict(t=50, b=80, l=65, r=50),
        )
        return fig

    z_vals, z_text, hover = [], [], []
    for cat in COT_CATS:
        row_z, row_t, row_h = [], [], []
        for comm in COT_COMMODITIES:
            d = HEATMAP_DATA.get(comm, {})
            bk_data = d.get('current', {}).get(bkdn) or d.get('current', {}).get('ALL', {})
            cat_data = bk_data.get(cat, {})
            m_data = cat_data.get(metric, {'v': 0, 'r': 50})
            v, r = m_data.get('v', 0) or 0, m_data.get('r', 50) or 50

            if colorby == 'rank':
                row_z.append(r)
            else:
                row_z.append(v)

            # Text in cell
            if metric in ('Net','Long','Short'):
                av = abs(v)
                txt = f"{'−' if v<0 else ''}{av/1e6:.1f}M" if av>=1e6 else \
                      f"{'−' if v<0 else ''}{av/1e3:.0f}K" if av>=1e3 else f"{v:.0f}"
            else:
                txt = f"{v:+.1f}%"
            row_t.append(txt)
            row_h.append(f"<b>{comm} — {COT_CAT_LABELS[cat]}</b><br>"
                         f"{metric}: {txt}<br>Percentile: {r}th")
        z_vals.append(row_z)
        z_text.append(row_t)
        hover.append(row_h)

    if colorby == 'rank':
        colorscale = [
            [0.00, '#A32D2D'], [0.15, '#E24B4A'], [0.25, '#F09595'],
            [0.35, '#FAEEDA'], [0.50, '#2D3348'], [0.65, '#C0DD97'],
            [0.75, '#56C271'], [0.85, '#1D9E75'], [1.00, '#085041'],
        ]
        zmin, zmax = 0, 100
    else:
        colorscale = 'RdYlGn'
        all_v = [v for row in z_vals for v in row if v is not None]
        zabs = max(abs(min(all_v)), abs(max(all_v))) if all_v else 30
        zmin, zmax = -zabs, zabs

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=COT_COMMODITIES,
        y=[COT_CAT_LABELS[c] for c in COT_CATS],
        hovertext=[[f"{COT_COMMODITIES[ci]} — {COT_CAT_LABELS[COT_CATS[ri]]}\n"
                    f"{z_text[ri][ci]}  |  {z_vals[ri][ci]:.0f}th pctile"
                    for ci in range(len(COT_COMMODITIES))]
                   for ri in range(len(COT_CATS))],
        hovertemplate='%{hovertext}<extra></extra>',
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(
            title=dict(text='Pctile rank' if colorby=='rank' else 'Value',
                       font=dict(size=10, color=P['sub'])),
            tickfont=dict(size=9, color=P['sub']),
            thickness=12, len=0.8,
        ),
        xgap=2, ygap=2,
    ))
    # Cell text via layout annotations (all Plotly versions)
    # Outline highlighted commodity
    if highlight and highlight in COT_COMMODITIES:
        xi = COT_COMMODITIES.index(highlight)
        fig.add_shape(type='rect',
            x0=xi-0.5, x1=xi+0.5, y0=-0.5, y1=len(COT_CATS)-0.5,
            line=dict(color=P['blue'], width=2.5),
        )

    fig.update_layout(
        height=360, paper_bgcolor=P['bg'], plot_bgcolor=P['panel'],
        font=dict(family='Courier New, monospace', color=P['text'], size=11),
        legend=dict(bgcolor='rgba(23,27,38,.9)', bordercolor=P['border'],
                    borderwidth=1, font=dict(size=10, color=P['sub']),
                    orientation='h', y=-0.22, x=0),
        margin=dict(t=40, b=100, l=160, r=60),
        hoverlabel=dict(bgcolor=P['panel'], bordercolor=P['border'],
                        font=dict(color=P['text'], size=11)),
        annotations=[
            dict(x=COT_COMMODITIES[ci], y=COT_CAT_LABELS[COT_CATS[ri]],
                 text=z_text[ri][ci], showarrow=False,
                 font=dict(size=10, color='white', family='Courier New, monospace'))
            for ri in range(len(COT_CATS)) for ci in range(len(COT_COMMODITIES))
        ]
    )
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=10, color=P['sub']))
    fig.update_yaxes(tickfont=dict(size=10, color=P['sub']))
    style_axes(fig)
    return fig


def fig_cot_history(comm='Cotton'):
    """MM net % and producer net % history for a given commodity."""
    hist = (MULTI_COT.get(comm) or {}).get('history', {})
    dates = hist.get('dates', COT_DATES)
    mm    = hist.get('MM',   COT_MM)
    prod  = hist.get('Prod', COT_PROD)
    oi    = hist.get('OI',   COT_OI)

    # If no time series data available, return informative empty chart
    if not dates:
        fig = go.Figure()
        fig.update_layout(
            **base_layout(460),
            title=dict(text=f'{comm} — historical positioning (add cotton_cot_clean.csv to load)',
                      font=dict(size=12, color=P['sub'])),
        )
        return style_axes(fig)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f'{comm} — net positioning % OI', 'Open interest (K contracts)'],
    )
    # 20/80 threshold bands
    d = HEATMAP_DATA.get(comm, {})
    ext = (d.get('current', {}).get('ALL') or {}).get('MM', {})
    mm_v = ext.get('NetPct', {}).get('v', 0)
    mm_r = ext.get('NetPct', {}).get('r', 50)

    fig.add_trace(go.Scatter(
        x=dates, y=[mm_v]*len(dates),
        mode='lines', line=dict(color=P['amber'], width=1, dash='dot'),
        name=f'Current MM ({mm_v:+.1f}% / {mm_r}th pctile)',
        hoverinfo='skip',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=mm, mode='lines',
        line=dict(color=P['blue'], width=2),
        name='MM net %', fill='tozeroy',
        fillcolor='rgba(75,147,209,.07)',
        hovertemplate='MM: %{y:.1f}%<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=prod, mode='lines',
        line=dict(color=P['coral'], width=1.5, dash='dash'),
        name='Producer net %',
        hovertemplate='Prod: %{y:.1f}%<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=oi,
        marker_color=P['muted'], marker_opacity=0.5,
        name='Open interest (K)',
        hovertemplate='OI: %{y:,}K<extra></extra>',
    ), row=2, col=1)
    fig.update_layout(**base_layout(460))
    style_axes(fig)
    fig.update_yaxes(title_text='Net % of OI', row=1, col=1)
    fig.update_yaxes(title_text='K contracts', row=2, col=1)
    return fig


def fig_cot_projections(comm='Cotton'):
    """
    Probability fan chart and analog forward paths for a commodity.
    """
    analogs  = COT_ANALOGS.get(comm, [])
    px_now   = COT_PX.get(comm, 100)
    unit     = COT_UNIT.get(comm, '')
    mm_r     = (HEATMAP_DATA.get(comm, {})
                .get('current', {})
                .get('ALL', {})
                .get('MM', {})
                .get('NetPct', {})
                .get('r', 50))

    base = BASE_RATES['short'] if mm_r<=20 else \
           BASE_RATES['long']  if mm_r>=80 else BASE_RATES['mid']

    dr = COT_DROUGHT.get(comm, None) if comm in COT_AG else None

    def drought_adj(hk):
        if not dr: return 0
        d2 = dr['d2']
        if hk in ('M1','M3'):
            return -6 if d2>=40 else -3 if d2>=20 else 0
        if hk in ('M6','Y1'):
            return  5 if d2>=40 else  2 if d2>=20 else 0
        return 0

    hkeys   = ['M1','M3','M6','Y1','Y18']
    h_lbls  = ['+1M','+3M','+6M','+1Y','+18M']
    h_dates = ['~Apr 17','~Jun 17','~Sep 17','~Mar 27','~Sep 27']

    avgs, meds, mins, maxs, probs_lo, probs_hi = [], [], [], [], [], []
    avg_pxs, bear_pxs, bull_pxs = [], [], []

    for hk in hkeys:
        rets = [a['fwd'][hk] for a in analogs if hk in a['fwd']]
        if not rets:
            avgs.append(0); meds.append(0); mins.append(0); maxs.append(0)
            probs_lo.append(50); probs_hi.append(50)
            avg_pxs.append(px_now); bear_pxs.append(px_now); bull_pxs.append(px_now)
            continue
        av  = float(np.mean(rets))
        md  = float(np.median(rets))
        mn  = min(rets); mx = max(rets)
        nb  = sum(1 for r in rets if r>0)
        da  = drought_adj(hk)
        plo = max(5, min(95, round(nb/len(rets)*100 + da)))
        phi = max(5, min(95, base[hk] + da))
        if plo > phi: plo, phi = phi, plo
        avgs.append(av); meds.append(md); mins.append(mn); maxs.append(mx)
        probs_lo.append(plo); probs_hi.append(phi)
        avg_pxs.append(round(px_now*(1+av/100), 1))
        bear_pxs.append(round(px_now*(1+mn/100), 1))
        bull_pxs.append(round(px_now*(1+mx/100), 1))

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=[f'{comm} — price projection fan  ({unit})',
                        f'Bull probability % by horizon'])

    # Fan: bear/bull range as filled band
    fig.add_trace(go.Scatter(
        x=h_lbls + h_lbls[::-1],
        y=bull_pxs + bear_pxs[::-1],
        fill='toself',
        fillcolor='rgba(75,147,209,.12)',
        line=dict(width=0),
        name='Analog range',
        hoverinfo='skip',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=h_lbls, y=avg_pxs,
        mode='lines+markers',
        line=dict(color=P['blue'], width=2.5),
        marker=dict(size=7, color=P['blue']),
        name='Avg projection',
        hovertemplate='%{x}: %{y}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=h_lbls, y=bull_pxs,
        mode='lines+markers',
        line=dict(color=P['green'], width=1.2, dash='dash'),
        marker=dict(size=5),
        name='Bull case',
        hovertemplate='Bull: %{y}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=h_lbls, y=bear_pxs,
        mode='lines+markers',
        line=dict(color=P['red'], width=1.2, dash='dash'),
        marker=dict(size=5),
        name='Bear case',
        hovertemplate='Bear: %{y}<extra></extra>',
    ), row=1, col=1)
    # Current price reference line
    fig.add_hline(y=px_now, line_color=P['muted'], line_dash='dot', line_width=1, row=1, col=1)

    # Probability bars
    mid_probs = [round((lo+hi)/2) for lo, hi in zip(probs_lo, probs_hi)]
    bar_cols   = [P['teal'] if p>=60 else P['green'] if p>=50 else P['amber'] if p>=40 else P['red']
                  for p in mid_probs]
    fig.add_trace(go.Bar(
        x=h_lbls, y=probs_hi,
        marker_color=[
            'rgba(29,158,117,0.2)' if c==P['teal'] else
            'rgba(86,194,113,0.2)' if c==P['green'] else
            'rgba(239,159,39,0.2)' if c==P['amber'] else
            'rgba(226,75,74,0.2)'
            for c in bar_cols],
        name='Broad base rate',
        hovertemplate='Base rate: %{y}%<extra></extra>',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=h_lbls, y=probs_lo,
        marker_color=bar_cols,
        name='Analog bull rate',
        hovertemplate='Analog: %{y}%<extra></extra>',
    ), row=1, col=2)
    fig.add_hline(y=50, line_color=P['muted'], line_dash='dot', line_width=1, row=1, col=2)

    fig.update_layout(**base_layout(420, barmode='overlay'))
    style_axes(fig)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text='% of analogs bullish', range=[0,100], row=1, col=2)
    return fig


def fig_cot_analogs(comm='Cotton'):
    """Forward return paths from each analog — normalized to current price."""
    analogs = COT_ANALOGS.get(comm, [])
    px_now  = COT_PX.get(comm, 100)
    unit    = COT_UNIT.get(comm, '')
    hkeys   = ['M1','M3','M6','Y1','Y18']
    h_lbls  = ['Now','+1M','+3M','+6M','+1Y','+18M']
    acols   = [P['red'],P['amber'],P['purple'],P['teal'],P['coral']]

    fig = go.Figure()
    for i, a in enumerate(analogs):
        scale = px_now / a['entry'] if a['entry'] else 1
        pts   = [px_now] + [round(a['fwd'][h]*scale*a['entry']/100 + px_now, 1)
                             if h in a['fwd'] else None for h in hkeys]
        fig.add_trace(go.Scatter(
            x=h_lbls, y=pts,
            mode='lines+markers',
            line=dict(color=acols[i], width=2),
            marker=dict(size=6, color=acols[i]),
            name=f"{a['yr']} ({a['match']}% match)",
            hovertemplate=f"{a['yr']}: %{{y:.1f}} {unit}<extra></extra>",
        ))

    fig.add_hline(y=px_now, line_color=P['muted'], line_dash='dot', line_width=1)
    fig.add_annotation(x='Now', y=px_now,
        text=f'Current: {px_now} {unit}',
        showarrow=False, yshift=12,
        font=dict(size=10, color=P['amber']),
    )
    fig.update_layout(
        **base_layout(400),
        title=dict(text=f'{comm} — analog forward paths (scaled to {px_now} {unit} entry)',
                   font=dict(size=12, color=P['sub'])),
    )
    style_axes(fig)
    fig.update_yaxes(title_text=unit)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi(label, value, sub='', color=P['text']):
    return html.Div([
        html.Div(label, style={'fontSize':'11px','color':P['sub'],'marginBottom':'4px'}),
        html.Div(value, style={'fontSize':'22px','fontWeight':'500','color':color}),
        html.Div(sub,   style={'fontSize':'11px','color':P['sub'],'marginTop':'3px'}),
    ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}',
              'borderRadius':'8px','padding':'12px 16px','flex':'1','minWidth':'150px'})


def dd(id_, opts, val, style_extra=None):
    s = {'fontSize':'14px','fontWeight':'700','minWidth':'200px','backgroundColor':P['panel'],'color':'#111'}
    if style_extra:
        s.update(style_extra)
    return dcc.Dropdown(id=id_, options=opts, value=val, clearable=False, style=s)


def section_header(text):
    return html.Div(text, style={
        'fontSize':'12px','fontWeight':'600','color':P['sub'],
        'borderBottom':f'0.5px solid {P["border"]}','paddingBottom':'6px',
        'marginBottom':'10px','marginTop':'6px',
        'textTransform':'uppercase','letterSpacing':'0.5px',
    })


def info_pill(text, color=P['blue']):
    return html.Span(text, style={
        'fontSize':'10px','fontWeight':'600','color':'#fff',
        'background':color,'borderRadius':'3px','padding':'2px 7px',
        'marginRight':'6px', 'display':'inline-block',
    })


def rank_badge(r):
    col = rank_col(r)
    lbl = 'Hist. short' if r<=15 else 'Below avg' if r<=30 else \
          'Above avg'   if r>=70 else 'Neutral'
    return html.Span(f'{r}th pctile — {lbl}', style={
        'fontSize':'10px','fontWeight':'600','color':'#fff',
        'background':col,'borderRadius':'3px','padding':'2px 8px',
    })

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 – ABOUT PAGE CONTENT (static)
# ══════════════════════════════════════════════════════════════════════════════

ABOUT_CONTENT = html.Div([
    # Hero
    html.Div([
        html.Div([
            html.Span('COT Multi-Commodity Positioning Dashboard',
                      style={'fontSize':'18px','fontWeight':'600','color':P['text']}),
            html.Span(' DISAGGREGATED', style={
                'fontSize':'10px','fontWeight':'600','color':P['blue'],
                'background':'rgba(75,147,209,.15)','border':f'1px solid {P["blue"]}',
                'borderRadius':'3px','padding':'2px 7px','marginLeft':'10px',
            }),
        ], style={'marginBottom':'8px'}),
        html.Div(
            'Aggregates 20 years of CFTC Disaggregated COT data (June 2006 – March 2026) '
            'across 14 futures markets. Identifies historically extreme positioning, matches '
            'current conditions to the closest analog weeks, and generates probability-weighted '
            'price projections at five forward horizons. For agricultural commodities, current '
            'USDA Drought Monitor readings and NOAA seasonal outlooks are incorporated as an '
            'overlay weight on top of the COT signals.',
            style={'fontSize':'12px','color':P['sub'],'lineHeight':'1.7'},
        ),
    ], style={'background':'linear-gradient(135deg,#1e3a5f,#2d5a8e)',
              'borderRadius':'8px','padding':'18px 22px','marginBottom':'16px'}),

    # Stats row
    html.Div([
        kpi('Weeks of COT history','928','June 2006 – March 2026',P['blue']),
        kpi('Commodity markets','14','Grains, softs, energy, metals, livestock',P['green']),
        kpi('Trader categories','5','MM / Producer / Swap / Other / Small Spec',P['amber']),
        kpi('Forward horizons','5','+1M  +3M  +6M  +1Y  +18M',P['purple']),
    ], style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'16px'}),

    section_header('How to use the COT tabs'),

    html.Div([
        html.Div([
            html.Div('② Positioning Heatmap', style={'fontSize':'12px','fontWeight':'600',
                                                      'color':P['text'],'marginBottom':'5px'}),
            html.Div([
                html.Li('Select Metric: Net % OI shows the contrarian signal most clearly. '
                        'Long % and Short % show one-sided pressure.'),
                html.Li('Select Contract Type: All = full picture; Old-crop = nearby marketing year; '
                        'Other-crop = deferred new-crop. Divergence between old and other often signals '
                        'a calendar spread trade or crop-year repricing.'),
                html.Li('Color by rank highlights extremes relative to that commodity\'s own history. '
                        'Deep red = historically short (≤15th pctile) = contrarian bull setup. '
                        'Deep green = historically long (≥85th pctile) = crowded, potential fade.'),
                html.Li('Highlight Row isolates one commodity so you can scan all 5 categories at once.'),
                html.Li('Click any bar in the chart to see the full detail panel for that cell.'),
            ], style={'fontSize':'11px','color':P['sub'],'lineHeight':'1.8','paddingLeft':'16px'}),
        ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}',
                  'borderRadius':'8px','padding':'14px','flex':'1'}),
        html.Div([
            html.Div('③ Price Projections', style={'fontSize':'12px','fontWeight':'600',
                                                    'color':P['text'],'marginBottom':'5px'}),
            html.Div([
                html.Li('Select a commodity from the dropdown — all charts update.'),
                html.Li('Signal pills at top show active extreme readings. Hover for explanation.'),
                html.Li('Drought overlay box appears for agricultural commodities only, '
                        'using USDA Drought Monitor D2+/D3+ coverage as of March 17, 2026.'),
                html.Li('Projection fan chart: shaded band = min/max analog range. '
                        'Blue line = avg projection. Dashed lines = bull/bear cases.'),
                html.Li('Probability bars: dark bar = analog-only bull rate (lower bound). '
                        'Light bar = broad base rate for all weeks in the same MM pctile zone (upper bound).'),
                html.Li('Analog paths chart shows each of the 5 closest historical weeks '
                        'normalized to today\'s price, so paths are directly comparable.'),
            ], style={'fontSize':'11px','color':P['sub'],'lineHeight':'1.8','paddingLeft':'16px'}),
        ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}',
                  'borderRadius':'8px','padding':'14px','flex':'1'}),
    ], style={'display':'flex','gap':'12px','marginBottom':'16px'}),

    section_header('Key terms glossary'),

    html.Div([
        html.Div([
            html.Div(term, style={'fontSize':'11px','fontWeight':'600','color':P['text'],'marginBottom':'3px'}),
            html.Div(defn, style={'fontSize':'11px','color':P['sub'],'lineHeight':'1.55'}),
        ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}',
                  'borderRadius':'6px','padding':'11px 13px'})
        for term, defn in [
            ('Managed Money (MM)',
             'Hedge funds and CTAs. Primary speculative category and strongest contrarian indicator. '
             'When ≤20th pctile, the broad 1-year forward return averages +17% with a 72% hit rate. '
             'When ≥80th pctile, 18-month returns turn negative as the crowded trade mean-reverts.'),
            ('Producer / Merchant',
             'Farmers, grain elevators, processors — physical hedgers. Structurally net short. '
             'When less short than normal (high pctile rank), they expect higher prices and are '
             'delaying hedges — a bullish read. Heavy hedging (low rank) implies expectation of lower prices.'),
            ('Swap Dealers',
             'Banks and financial intermediaries managing OTC derivatives books, typically net long '
             'in futures to offset short commodity swaps sold to clients. Extremes here reflect '
             'structural institutional demand rather than directional bets.'),
            ('Other Reportable',
             'Index funds, pension allocations, CTA-adjacent. A 100th pctile reading means maximum '
             'structural allocation — near-term crowding risk but also long-term demand support. '
             'Watch for any unwind as a risk-off signal.'),
            ('Small Speculators',
             'Retail traders below CFTC reporting thresholds. Generally contrarian — retail at '
             '≥90th pctile is bearish; retail at ≤10th pctile (capitulation) often marks a bottom '
             'alongside extreme MM positioning.'),
            ('Old-crop vs Other-crop',
             'Old-crop = nearby futures tied to the current marketing year (already planted or harvested). '
             'Other-crop = deferred contracts for the next crop year. Divergence often signals a '
             'crop-year transition, roll trade, or different supply/demand expectations across time horizons.'),
            ('Percentile rank',
             'Where current positioning sits vs every historical week for that specific commodity. '
             '100th = highest ever recorded; 0th = lowest ever. Each commodity is ranked '
             'independently — a 20th pctile in Cotton ≠ a 20th pctile in WTI Crude.'),
            ('D2+ / D3+ drought zones',
             'USDA Drought Monitor: D2 = Severe, D3 = Extreme, D4 = Exceptional. '
             'D2+ coverage above 30-40% of major growing areas during planting is a significant '
             'supply-side headwind. D3+ above 15% is historically a strong predictor of below-trend '
             'yields. Used only for agricultural commodities. Source: dm_export_20060320_20260320.csv.'),
        ]
    ], style={'display':'grid','gridTemplateColumns':'repeat(2,1fr)','gap':'8px'}),
])

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 – DASH APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = Dash(__name__, title='Cotton & COT Dashboard')

_tab_style        = {'fontFamily':'Courier New, monospace','fontSize':'12px',
                     'color':P['sub'],'padding':'6px 14px'}
_tab_selected     = {**_tab_style,'color':P['text'],'borderTop':f'2px solid {P["blue"]}'}

app.layout = html.Div(
    style={'background':P['bg'],'minHeight':'100vh',
           'fontFamily':'Courier New, monospace','color':P['text'],'padding':'20px'},
    children=[

        # ── Master header ──────────────────────────────────────────────────────
        html.Div([
            html.Div('Cotton Belt Drought & COT Positioning Dashboard',
                     style={'fontSize':'22px','fontWeight':'500','marginBottom':'4px'}),
            html.Div('2026 season  ·  USDA Drought Monitor  ·  NCC Planting Intentions  ·  CBOT Futures  ·  CFTC COT (2006–2026)',
                     style={'fontSize':'12px','color':P['sub']}),
        ], style={'marginBottom':'16px'}),

        # ── Cotton KPI row ─────────────────────────────────────────────────────
        html.Div([
            kpi('NCC 2026 intended acres','8.83M','−3.4% vs 2025'),
            kpi('Baseline production', f"{NAT_BASE/1000:.1f}M bales",'drought-adjusted',P['amber']),
            kpi('Failure scenario',    f"{NAT_FAIL/1000:.1f}M bales",'<1" S.TX/CB by plant date',P['red']),
            kpi('Belt D2+ (Mar 2026)', '46%','avg area severe+ drought',P['red']),
            kpi('Cotton MM pctile',    '12th','historically short — contrarian bull',P['blue']),
            kpi('Sugar MM pctile',     '2nd', 'record short — strongest COT signal',P['red']),
        ], style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'20px'}),

        # ── Master tabs ────────────────────────────────────────────────────────
        dcc.Tabs(
            id='tabs', value='tab-prod',
            colors={'border':P['border'],'primary':P['blue'],'background':P['bg']},
            children=[
                dcc.Tab(label='Production & analogs',      value='tab-prod',
                        style=_tab_style, selected_style=_tab_selected),
                dcc.Tab(label='Seasonal drought profile',  value='tab-seas',
                        style=_tab_style, selected_style=_tab_selected),
                dcc.Tab(label='Futures & planting signal', value='tab-fut',
                        style=_tab_style, selected_style=_tab_selected),
                dcc.Tab(label='State detail',              value='tab-states',
                        style=_tab_style, selected_style=_tab_selected),
                dcc.Tab(label='COT — About & guide',       value='tab-cot-about',
                        style=_tab_style, selected_style=_tab_selected),
                dcc.Tab(label='COT — Positioning heatmap', value='tab-cot-heat',
                        style=_tab_style, selected_style=_tab_selected),
                dcc.Tab(label='COT — Price projections',   value='tab-cot-proj',
                        style=_tab_style, selected_style=_tab_selected),
            ],
        ),

        # ── Tab-specific controls (shown/hidden by callback) ───────────────────
        # Seasonal
        html.Div(id='wrap-seas', style={'display':'none','marginTop':'14px'}, children=[
            html.Div('Compare against:',
                     style={'fontSize':'12px','color':P['sub'],'marginBottom':'4px'}),
            dd('analog-sel',[
                {'label':'All analog years','value':'all'},
                {'label':'2011','value':'2011'},{'label':'2022','value':'2022'},
                {'label':'2007','value':'2007'},{'label':'2012','value':'2012'},
                {'label':'2013','value':'2013'},
            ],'all'),
        ]),

        # State controls
        html.Div(id='wrap-states', style={'display':'none','marginTop':'14px'}, children=[
            html.Div(style={'display':'flex','gap':'20px','flexWrap':'wrap'}, children=[
                html.Div([
                    html.Div('Scenario:',style={'fontSize':'12px','color':P['sub'],'marginBottom':'4px'}),
                    dd('scen-sel',[
                        {'label':'Baseline','value':'base'},
                        {'label':'S.TX/CB failure (<1" by plant date)','value':'fail'},
                    ],'base'),
                ]),
                html.Div([
                    html.Div('Sort by:',style={'fontSize':'12px','color':P['sub'],'marginBottom':'4px'}),
                    dd('sort-sel',[
                        {'label':'Production','value':'prod'},
                        {'label':'Planted acres','value':'acres'},
                        {'label':'Drought severity','value':'drought'},
                    ],'prod'),
                ]),
            ]),
        ]),

        # COT Heatmap controls
        html.Div(id='wrap-cot-heat', style={'display':'none','marginTop':'14px'}, children=[
            html.Div(style={'display':'flex','gap':'14px','flexWrap':'wrap','alignItems':'flex-end'}, children=[
                html.Div([
                    html.Div('Metric', style={'fontSize':'11px','color':P['sub'],'marginBottom':'4px'}),
                    dd('cot-metric',[
                        {'label':'Net % of open interest','value':'NetPct'},
                        {'label':'Long % of open interest','value':'LongPct'},
                        {'label':'Short % of open interest','value':'ShortPct'},
                        {'label':'Net contracts','value':'Net'},
                        {'label':'Long contracts','value':'Long'},
                        {'label':'Short contracts','value':'Short'},
                    ],'NetPct'),
                ]),
                html.Div([
                    html.Div('Contract type', style={'fontSize':'11px','color':P['sub'],'marginBottom':'4px'}),
                    dd('cot-bkdn',[
                        {'label':'All contracts','value':'ALL'},
                        {'label':'Old-crop (legacy)','value':'Old'},
                        {'label':'Other-crop (deferred)','value':'Oth'},
                    ],'ALL'),
                ]),
                html.Div([
                    html.Div('Color by', style={'fontSize':'11px','color':P['sub'],'marginBottom':'4px'}),
                    dd('cot-colorby',[
                        {'label':'Percentile rank vs history','value':'rank'},
                        {'label':'Raw value magnitude','value':'value'},
                    ],'rank'),
                ]),
                html.Div([
                    html.Div('Highlight commodity', style={'fontSize':'11px','color':P['sub'],'marginBottom':'4px'}),
                    dd('cot-hi',[{'label':'— None —','value':''}] +
                       [{'label':c,'value':c} for c in COT_COMMODITIES],''),
                ]),
            ]),
        ]),

        # COT Projections controls
        html.Div(id='wrap-cot-proj', style={'display':'none','marginTop':'14px'}, children=[
            html.Div([
                html.Div('Commodity', style={'fontSize':'11px','color':P['sub'],'marginBottom':'4px'}),
                dd('cot-comm',[{'label':c,'value':c} for c in ['Cotton','Corn','Soybeans','SRW Wheat','HRW Wheat','Sugar','WTI Crude','Gold','Silver','Live Cattle','Lean Hogs','Soy Oil','Soy Meal','Cocoa','Coffee']],'Cotton'),
            ], style={'display':'inline-block'}),
        ]),

        # ── Chart output area ──────────────────────────────────────────────────
        html.Div(id='chart-out', style={'marginTop':'14px'}),
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 – CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('wrap-seas',      'style'),
    Output('wrap-states',    'style'),
    Output('wrap-cot-heat',  'style'),
    Output('wrap-cot-proj',  'style'),
    Input('tabs', 'value'),
)
def show_controls(tab):
    show = {'display':'block','marginTop':'14px'}
    hide = {'display':'none'}
    return (
        show if tab=='tab-seas'      else hide,
        show if tab=='tab-states'    else hide,
        show if tab=='tab-cot-heat'  else hide,
        show if tab=='tab-cot-proj'  else hide,
    )


@app.callback(
    Output('chart-out', 'children'),
    Input('tabs',        'value'),
    # original controls
    Input('analog-sel',  'value'),
    Input('scen-sel',    'value'),
    Input('sort-sel',    'value'),
    # COT heatmap controls
    Input('cot-metric',  'value'),
    Input('cot-bkdn',    'value'),
    Input('cot-colorby', 'value'),
    Input('cot-hi',      'value'),
    # COT projection control
    Input('cot-comm',    'value'),
)
def render(tab, analog_sel, scen_sel, sort_sel,
           cot_metric, cot_bkdn, cot_colorby, cot_hi, cot_comm):

    cfg = dict(displayModeBar=True, scrollZoom=False,
               toImageButtonOptions=dict(format='png', scale=2))

    # ── Original tabs ──────────────────────────────────────────────────────────
    if tab == 'tab-prod':
        return dcc.Graph(figure=fig_production(), config=cfg)

    if tab == 'tab-seas':
        return dcc.Graph(figure=fig_seasonal(analog_sel or 'all'), config=cfg)

    if tab == 'tab-fut':
        return dcc.Graph(figure=fig_futures(), config=cfg)

    if tab == 'tab-states':
        return dcc.Graph(figure=fig_states(scen_sel or 'base', sort_sel or 'prod'), config=cfg)

    # ── COT About ─────────────────────────────────────────────────────────────
    if tab == 'tab-cot-about':
        return ABOUT_CONTENT

    # ── COT Heatmap ───────────────────────────────────────────────────────────
    if tab == 'tab-cot-heat':
        metric  = cot_metric  or 'NetPct'
        bkdn    = cot_bkdn    or 'ALL'
        colorby = cot_colorby or 'rank'
        hi      = cot_hi      or ''

        # Build signal summary cards for extremes
        extreme_cards = []
        if HEATMAP_DATA:
            for comm in COT_COMMODITIES:
                d    = HEATMAP_DATA.get(comm, {})
                bkd  = (d.get('current') or {}).get(bkdn) or \
                       (d.get('current') or {}).get('ALL', {})
                for cat in COT_CATS:
                    m = (bkd.get(cat) or {}).get(metric) or {}
                    r = m.get('r', 50)
                    if r <= 10 or r >= 90:
                        is_bull = r >= 90
                        col = P['teal'] if is_bull else P['red']
                        extreme_cards.append(html.Div([
                            html.Div([
                                html.Span(f"{comm}", style={'fontWeight':'600','color':col}),
                                html.Span(f" · {COT_CAT_LABELS[cat]}",
                                          style={'color':P['sub'],'fontSize':'10px'}),
                                html.Span(f" {r}th", style={
                                    'fontSize':'10px','fontWeight':'600','color':'#fff',
                                    'background':col,'borderRadius':'3px',
                                    'padding':'1px 5px','marginLeft':'5px',
                                }),
                            ], style={'marginBottom':'3px'}),
                            html.Div(
                                'Historically long — crowding risk' if is_bull
                                else 'Historically short — contrarian bull',
                                style={'fontSize':'10px','color':P['sub']},
                            ),
                        ], style={
                            'background':P['panel'],'border':f'0.5px solid {col}',
                            'borderRadius':'6px','padding':'8px 10px',
                        }))

        ext_section = html.Div([
            html.Div(
                f'Notable extremes — {len(extreme_cards)} cells at ≤10th or ≥90th percentile',
                style={'fontSize':'12px','fontWeight':'600','color':P['sub'],
                       'marginBottom':'8px','marginTop':'14px'},
            ),
            html.Div(extreme_cards,
                     style={'display':'grid','gridTemplateColumns':'repeat(auto-fill,minmax(200px,1fr))',
                             'gap':'8px'}),
        ]) if extreme_cards else html.Div()

        # COT history chart (uses selected highlighted commodity if any, else Cotton)
        hist_comm = hi if hi else 'Cotton'
        return html.Div([
            dcc.Graph(figure=fig_cot_heatmap(metric, bkdn, colorby, hi), config=cfg),
            ext_section,
            html.Div(style={'height':'16px'}),
            html.Div(f'Historical positioning — {hist_comm}',
                     style={'fontSize':'12px','fontWeight':'600','color':P['sub'],'marginBottom':'6px'}),
            dcc.Graph(figure=fig_cot_history(hist_comm), config=cfg),
        ])


    # ── COT Projections ───────────────────────────────────────────────────────
    if tab == 'tab-cot-proj':
        comm = cot_comm or 'Cotton'

        # Per-commodity metadata
        CMETA = {
            'Cotton':      {'unit':'¢/lb',  'px':74.03,  'ag':True,  'sector':'Softs',    'contract':'ICE Dec'},
            'Corn':        {'unit':'¢/bu',  'px':490,   'ag':True,  'sector':'Grains',   'contract':'CBOT Dec'},
            'Soybeans':    {'unit':'¢/bu',  'px':1144,   'ag':True,  'sector':'Grains',   'contract':'CBOT Nov'},
            'SRW Wheat':   {'unit':'¢/bu',  'px':542,   'ag':True,  'sector':'Grains',   'contract':'CBOT Jul'},
            'HRW Wheat':   {'unit':'¢/bu',  'px':565,   'ag':True,  'sector':'Grains',   'contract':'KCBT Jul'},
            'Sugar':       {'unit':'¢/lb',  'px':18.4,  'ag':True,  'sector':'Softs',    'contract':'ICE No.11'},
            'WTI Crude':   {'unit':'$/bbl', 'px':68.4,  'ag':False, 'sector':'Energy',   'contract':'NYMEX CL'},
            'Gold':        {'unit':'$/oz',  'px':3042,  'ag':False, 'sector':'Metals',   'contract':'COMEX Jun'},
            'Silver':      {'unit':'$/oz',  'px':33.8,  'ag':False, 'sector':'Metals',   'contract':'COMEX May'},
            'Live Cattle': {'unit':'¢/lb',  'px':196,   'ag':False, 'sector':'Livestock','contract':'CME Jun'},
            'Lean Hogs':   {'unit':'¢/lb',  'px':88,    'ag':False, 'sector':'Livestock','contract':'CME Jun'},
            'Soy Oil':     {'unit':'¢/lb',  'px':41.2,  'ag':True,  'sector':'Grains',   'contract':'CBOT Jul'},
            'Soy Meal':    {'unit':'$/ton', 'px':296,   'ag':True,  'sector':'Grains',   'contract':'CBOT Jul'},
            'Cocoa':       {'unit':'$/MT',  'px':8840,  'ag':True,  'sector':'Softs',    'contract':'ICE Jul'},
            'Coffee':      {'unit':'¢/lb',  'px':374,   'ag':True,  'sector':'Softs',    'contract':'ICE Sep'},
        }
        meta     = CMETA.get(comm, {'unit':'','px':100,'ag':False,'sector':'','contract':''})
        unit     = meta['unit']
        px_now   = meta['px']
        is_ag    = meta['ag']
        sector   = meta['sector']

        # Bloomberg COT signal data
        BCOM = {
            'Cotton':      {'gs':99.5, 'net_s':99.1, 'gl_asc':40.6, 'short_oi':96.0, 'net_oi':97.3, 'long_oi_asc':10.3, 'mm_net':-72937, 'mm_long':41093,  'mm_short':114030, 'prod_net':-30351, 'swap_net':55819,   'other_net':42790},
            'Corn':        {'gs':62.1, 'net_s':57.5, 'gl_asc':59.2, 'short_oi':57.8, 'net_oi':59.1, 'long_oi_asc':45.1, 'mm_net':52974,  'mm_long':247478, 'mm_short':194504, 'prod_net':-333611,'swap_net':301263,  'other_net':8515},
            'Soybeans':    {'gs':54.6, 'net_s':4.8,  'gl_asc':98.4, 'short_oi':41.0, 'net_oi':21.7, 'long_oi_asc':86.4, 'mm_net':198902, 'mm_long':239399, 'mm_short':40497,  'prod_net':-279601,'swap_net':103244,  'other_net':15237},
            'SRW Wheat':   {'gs':52.0, 'net_s':40.0, 'gl_asc':88.4, 'short_oi':52.6, 'net_oi':39.7, 'long_oi_asc':87.8, 'mm_net':-25800, 'mm_long':94101,  'mm_short':119901, 'prod_net':-48774, 'swap_net':75373,   'other_net':963},
            'HRW Wheat':   {'gs':65.9, 'net_s':49.6, 'gl_asc':82.1, 'short_oi':52.3, 'net_oi':50.4, 'long_oi_asc':60.1, 'mm_net':1866,   'mm_long':69127,  'mm_short':67261,  'prod_net':-76601, 'swap_net':78832,   'other_net':317},
            'Sugar':       {'gs':100.0,'net_s':100.0,'gl_asc':41.0, 'short_oi':99.9, 'net_oi':99.8, 'long_oi_asc':13.5, 'mm_net':-248296,'mm_long':139593, 'mm_short':387889, 'prod_net':43355,  'swap_net':192066,  'other_net':16438},
            'WTI Crude':   {'gs':65.3, 'net_s':67.8, 'gl_asc':34.9, 'short_oi':34.1, 'net_oi':62.5, 'long_oi_asc':21.9, 'mm_net':-10570, 'mm_long':19343,  'mm_short':29913,  'prod_net':75801,  'swap_net':-113613, 'other_net':47835},
            'Gold':        {'gs':35.9, 'net_s':59.1, 'gl_asc':28.5, 'short_oi':38.3, 'net_oi':62.6, 'long_oi_asc':18.1, 'mm_net':100855, 'mm_long':124006, 'mm_short':23151,  'prod_net':-26027, 'swap_net':-177784, 'other_net':59037},
            'Silver':      {'gs':11.3, 'net_s':75.3, 'gl_asc':0.5,  'short_oi':11.7, 'net_oi':74.2, 'long_oi_asc':0.7,  'mm_net':9721,   'mm_long':13102,  'mm_short':3381,   'prod_net':-16149, 'swap_net':-24701,  'other_net':14015},
            'Live Cattle': {'gs':23.2, 'net_s':13.9, 'gl_asc':88.1, 'short_oi':19.0, 'net_oi':18.2, 'long_oi_asc':80.5, 'mm_net':114519, 'mm_long':128696, 'mm_short':14177,  'prod_net':-160073,'swap_net':59736,   'other_net':7437},
            'Lean Hogs':   {'gs':27.3, 'net_s':1.5,  'gl_asc':98.7, 'short_oi':7.4,  'net_oi':3.2,  'long_oi_asc':95.4, 'mm_net':124036, 'mm_long':137543, 'mm_short':13507,  'prod_net':-184955,'swap_net':70325,   'other_net':8854},
            'Soy Oil':     {'gs':73.5, 'net_s':11.5, 'gl_asc':100.0,'short_oi':46.3, 'net_oi':37.1, 'long_oi_asc':64.4, 'mm_net':75509,  'mm_long':141663, 'mm_short':66154,  'prod_net':-152612,'swap_net':74379,   'other_net':-9639},
            'Soy Meal':    {'gs':65.6, 'net_s':9.7,  'gl_asc':95.5, 'short_oi':45.3, 'net_oi':37.9, 'long_oi_asc':67.1, 'mm_net':80661,  'mm_long':115332, 'mm_short':34671,  'prod_net':-208373,'swap_net':92776,   'other_net':12312},
            'Cocoa':       {'gs':69.2, 'net_s':82.0, 'gl_asc':12.4, 'short_oi':66.6, 'net_oi':81.1, 'long_oi_asc':3.1,  'mm_net':-7921,  'mm_long':28263,  'mm_short':36184,  'prod_net':-19344, 'swap_net':35119,   'other_net':-9436},
            'Coffee':      {'gs':58.2, 'net_s':51.2, 'gl_asc':46.6, 'short_oi':58.3, 'net_oi':50.5, 'long_oi_asc':61.3, 'mm_net':12363,  'mm_long':35131,  'mm_short':22768,  'prod_net':-17181, 'swap_net':5820,    'other_net':-888},
        }
        bcom        = BCOM.get(comm, {})
        gs          = bcom.get('gs',    50.0)
        net_s       = bcom.get('net_s', 50.0)
        gl_asc      = bcom.get('gl_asc',50.0)
        net_oi      = bcom.get('net_oi',50.0)
        long_oi_asc = bcom.get('long_oi_asc', 50.0)
        short_oi    = bcom.get('short_oi', 50.0)

        # COT bull score 0-100
        bull_score = round(0.35*gs + 0.35*net_s + 0.20*(100-gl_asc) + 0.10*(100-long_oi_asc))
        if   bull_score >= 85: cot_signal, cot_col, cot_icon = 'STRONG BUY',  P['teal'],  '▲▲'
        elif bull_score >= 65: cot_signal, cot_col, cot_icon = 'BUY',          P['green'], '▲'
        elif bull_score <= 15: cot_signal, cot_col, cot_icon = 'STRONG SELL',  P['red'],   '▼▼'
        elif bull_score <= 35: cot_signal, cot_col, cot_icon = 'SELL',          P['amber'], '▼'
        else:                  cot_signal, cot_col, cot_icon = 'NEUTRAL',       P['muted'], '–'

        # Analog returns per commodity
        ANALOGS = {
            'Cotton':      [{'yr':'2024 May','match':91,'entry':76.2, 'fwd':{'M1':-3.9,'M3':-6.1,'M6':-9.7, 'Y1':-15.8}},
                            {'yr':'2019 May','match':84,'entry':68.2, 'fwd':{'M1':-3.0,'M3':-11.5,'M6':-7.8,'Y1':-14.0}},
                            {'yr':'2020 Mar','match':79,'entry':54.6, 'fwd':{'M1':4.2, 'M3':12.1,'M6':20.5, 'Y1':61.8}},
                            {'yr':'2019 Mar','match':77,'entry':73.4, 'fwd':{'M1':-3.1,'M3':-9.9,'M6':-19.8,'Y1':-19.1}},
                            {'yr':'2015 Sep','match':74,'entry':63.8, 'fwd':{'M1':2.4, 'M3':5.8, 'M6':8.4,  'Y1':12.8}}],
            'Corn':        [{'yr':'2023 Sep','match':85,'entry':484,  'fwd':{'M1':-2.1,'M3':-4.8,'M6':3.2,  'Y1':5.4}},
                            {'yr':'2016 Jun','match':79,'entry':348,  'fwd':{'M1':4.2, 'M3':8.6, 'M6':14.2, 'Y1':22.4}},
                            {'yr':'2014 Sep','match':75,'entry':322,  'fwd':{'M1':-1.8,'M3':-2.4,'M6':4.8,  'Y1':8.4}},
                            {'yr':'2019 Mar','match':72,'entry':368,  'fwd':{'M1':-3.2,'M3':-6.4,'M6':2.4,  'Y1':6.8}},
                            {'yr':'2021 Jun','match':68,'entry':548,  'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-22.4}}],
            'Soybeans':    [{'yr':'2022 Jun','match':86,'entry':1484, 'fwd':{'M1':-4.2,'M3':-8.6,'M6':-12.4,'Y1':-18.4}},
                            {'yr':'2020 Jun','match':80,'entry':862,  'fwd':{'M1':2.4, 'M3':8.4, 'M6':18.4, 'Y1':32.4}},
                            {'yr':'2018 Mar','match':76,'entry':1042, 'fwd':{'M1':-2.8,'M3':-6.4,'M6':-8.4, 'Y1':-12.4}},
                            {'yr':'2017 Jun','match':72,'entry':974,  'fwd':{'M1':-1.8,'M3':-4.2,'M6':2.4,  'Y1':8.4}},
                            {'yr':'2021 Sep','match':68,'entry':1228, 'fwd':{'M1':-3.2,'M3':-8.4,'M6':-14.4,'Y1':-18.4}}],
            'SRW Wheat':   [{'yr':'2022 Mar','match':85,'entry':784,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':2.4,  'Y1':-12.4}},
                            {'yr':'2018 Sep','match':80,'entry':512,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4}},
                            {'yr':'2020 Sep','match':76,'entry':498,  'fwd':{'M1':2.4, 'M3':6.4, 'M6':12.4, 'Y1':24.4}},
                            {'yr':'2016 Mar','match':72,'entry':484,  'fwd':{'M1':-1.8,'M3':2.4, 'M6':8.4,  'Y1':14.4}},
                            {'yr':'2019 Jun','match':68,'entry':492,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4}}],
            'HRW Wheat':   [{'yr':'2022 Mar','match':84,'entry':820,  'fwd':{'M1':8.4, 'M3':12.4,'M6':-2.4, 'Y1':-14.4}},
                            {'yr':'2020 Sep','match':78,'entry':520,  'fwd':{'M1':3.2, 'M3':8.4, 'M6':14.4, 'Y1':28.4}},
                            {'yr':'2016 Mar','match':74,'entry':448,  'fwd':{'M1':-2.4,'M3':4.8, 'M6':12.4, 'Y1':18.4}},
                            {'yr':'2019 Jun','match':70,'entry':490,  'fwd':{'M1':-1.8,'M3':-3.2,'M6':4.8,  'Y1':12.4}},
                            {'yr':'2018 Jun','match':66,'entry':534,  'fwd':{'M1':-3.2,'M3':-6.4,'M6':2.4,  'Y1':8.4}}],
            'Sugar':       [{'yr':'2015 Sep','match':92,'entry':10.8, 'fwd':{'M1':8.4, 'M3':18.4,'M6':28.4, 'Y1':42.4}},
                            {'yr':'2019 Jun','match':86,'entry':11.4, 'fwd':{'M1':4.2, 'M3':12.4,'M6':24.4, 'Y1':18.4}},
                            {'yr':'2020 Mar','match':82,'entry':12.2, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':18.4, 'Y1':32.4}},
                            {'yr':'2018 Sep','match':78,'entry':13.8, 'fwd':{'M1':-2.4,'M3':-4.8,'M6':4.8,  'Y1':12.4}},
                            {'yr':'2011 Jun','match':74,'entry':26.4, 'fwd':{'M1':-8.4,'M3':-18.4,'M6':-28.4,'Y1':-38.4}}],
            'WTI Crude':   [{'yr':'2023 Mar','match':88,'entry':76.8, 'fwd':{'M1':-4.8,'M3':-8.4,'M6':-12.4,'Y1':-18.4}},
                            {'yr':'2019 Jun','match':82,'entry':64.2, 'fwd':{'M1':-2.4,'M3':-6.4,'M6':4.8,  'Y1':8.4}},
                            {'yr':'2020 Sep','match':78,'entry':42.4, 'fwd':{'M1':8.4, 'M3':18.4,'M6':24.4, 'Y1':28.4}},
                            {'yr':'2016 Mar','match':74,'entry':34.8, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':22.4}},
                            {'yr':'2018 Dec','match':70,'entry':52.4, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':24.4}}],
            'Gold':        [{'yr':'2020 Sep','match':88,'entry':1912, 'fwd':{'M1':4.8, 'M3':8.4, 'M6':4.8,  'Y1':-8.4}},
                            {'yr':'2022 Mar','match':82,'entry':1912, 'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-8.4}},
                            {'yr':'2023 Sep','match':78,'entry':1924, 'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':22.4}},
                            {'yr':'2019 Sep','match':74,'entry':1484, 'fwd':{'M1':2.4, 'M3':4.8, 'M6':8.4,  'Y1':18.4}},
                            {'yr':'2021 Jun','match':70,'entry':1764, 'fwd':{'M1':-2.4,'M3':-8.4,'M6':-4.8, 'Y1':-12.4}}],
            'Silver':      [{'yr':'2020 Mar','match':88,'entry':14.4, 'fwd':{'M1':8.4, 'M3':24.4,'M6':42.4, 'Y1':38.4}},
                            {'yr':'2016 Mar','match':82,'entry':14.8, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':14.4}},
                            {'yr':'2019 Sep','match':78,'entry':18.2, 'fwd':{'M1':-2.4,'M3':4.8, 'M6':12.4, 'Y1':8.4}},
                            {'yr':'2022 Sep','match':74,'entry':18.4, 'fwd':{'M1':-4.8,'M3':-8.4,'M6':-4.8, 'Y1':4.8}},
                            {'yr':'2023 Jun','match':70,'entry':22.4, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':12.4, 'Y1':24.4}}],
            'Live Cattle': [{'yr':'2022 Mar','match':86,'entry':144,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':8.4}},
                            {'yr':'2014 Jun','match':80,'entry':148,  'fwd':{'M1':2.4, 'M3':4.8, 'M6':-4.8, 'Y1':-18.4}},
                            {'yr':'2023 Sep','match':76,'entry':184,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':-8.4, 'Y1':-4.8}},
                            {'yr':'2021 Jun','match':72,'entry':122,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':14.4, 'Y1':18.4}},
                            {'yr':'2019 Mar','match':68,'entry':132,  'fwd':{'M1':2.4, 'M3':4.8, 'M6':8.4,  'Y1':12.4}}],
            'Lean Hogs':   [{'yr':'2021 Jun','match':87,'entry':112,  'fwd':{'M1':-8.4,'M3':-18.4,'M6':-28.4,'Y1':-42.4}},
                            {'yr':'2014 Mar','match':82,'entry':124,  'fwd':{'M1':-12.4,'M3':-22.4,'M6':-32.4,'Y1':-38.4}},
                            {'yr':'2022 Mar','match':78,'entry':104,  'fwd':{'M1':-2.4,'M3':-8.4,'M6':-18.4,'Y1':-24.4}},
                            {'yr':'2023 Mar','match':74,'entry':88,   'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':18.4}},
                            {'yr':'2020 Mar','match':70,'entry':62,   'fwd':{'M1':-8.4,'M3':-12.4,'M6':4.8,  'Y1':18.4}}],
            'Soy Oil':     [{'yr':'2021 Sep','match':85,'entry':62.4, 'fwd':{'M1':-2.4,'M3':-8.4,'M6':-14.4,'Y1':-22.4}},
                            {'yr':'2022 Jun','match':80,'entry':72.4, 'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-24.4}},
                            {'yr':'2020 Jun','match':76,'entry':28.4, 'fwd':{'M1':4.8, 'M3':12.4,'M6':18.4, 'Y1':28.4}},
                            {'yr':'2018 Mar','match':72,'entry':32.4, 'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4}},
                            {'yr':'2019 Sep','match':68,'entry':28.8, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':14.4, 'Y1':22.4}}],
            'Soy Meal':    [{'yr':'2022 Jun','match':84,'entry':424,  'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-28.4}},
                            {'yr':'2020 Sep','match':78,'entry':312,  'fwd':{'M1':2.4, 'M3':8.4, 'M6':18.4, 'Y1':28.4}},
                            {'yr':'2018 Mar','match':74,'entry':384,  'fwd':{'M1':-2.4,'M3':-4.8,'M6':2.4,  'Y1':8.4}},
                            {'yr':'2017 Sep','match':70,'entry':318,  'fwd':{'M1':4.8, 'M3':8.4, 'M6':12.4, 'Y1':18.4}},
                            {'yr':'2021 Jun','match':66,'entry':392,  'fwd':{'M1':-2.4,'M3':-8.4,'M6':-12.4,'Y1':-18.4}}],
            'Cocoa':       [{'yr':'2017 Mar','match':88,'entry':2142, 'fwd':{'M1':8.4, 'M3':18.4,'M6':38.4, 'Y1':48.4}},
                            {'yr':'2014 Sep','match':82,'entry':2842, 'fwd':{'M1':-4.8,'M3':-12.4,'M6':-18.4,'Y1':-28.4}},
                            {'yr':'2019 Sep','match':78,'entry':2384, 'fwd':{'M1':4.8, 'M3':12.4,'M6':22.4, 'Y1':38.4}},
                            {'yr':'2016 Mar','match':74,'entry':2724, 'fwd':{'M1':2.4, 'M3':8.4, 'M6':12.4, 'Y1':18.4}},
                            {'yr':'2023 Mar','match':70,'entry':2584, 'fwd':{'M1':12.4,'M3':28.4,'M6':48.4, 'Y1':62.4}}],
            'Coffee':      [{'yr':'2021 Sep','match':86,'entry':218,  'fwd':{'M1':-4.8,'M3':-8.4,'M6':-12.4,'Y1':-18.4}},
                            {'yr':'2022 Jun','match':80,'entry':238,  'fwd':{'M1':4.8, 'M3':12.4,'M6':22.4, 'Y1':38.4}},
                            {'yr':'2019 Mar','match':76,'entry':94,   'fwd':{'M1':4.8, 'M3':18.4,'M6':38.4, 'Y1':62.4}},
                            {'yr':'2018 Jun','match':72,'entry':118,  'fwd':{'M1':2.4, 'M3':8.4, 'M6':12.4, 'Y1':22.4}},
                            {'yr':'2023 Mar','match':68,'entry':182,  'fwd':{'M1':8.4, 'M3':18.4,'M6':28.4, 'Y1':42.4}}],
        }
        analogs = ANALOGS.get(comm, [])

        # Horizons
        HORIZONS = [('M1','+1 Month','~Apr 17 2026'),('M3','+3 Months','~Jun 17 2026'),
                    ('M6','+6 Months','~Sep 17 2026'),('Y1','+1 Year','~Mar 17 2027')]

        # Build projection cards
        proj_cards = []
        for hk, hl, hd in HORIZONS:
            rets = [a['fwd'][hk] for a in analogs if hk in a['fwd']]
            if not rets:
                continue
            av  = float(np.mean(rets))
            mn  = min(rets)
            mx  = max(rets)
            nb  = sum(1 for r in rets if r > 0)
            pct = round(nb / len(rets) * 100)
            # COT adjustment
            cot_adj = round((bull_score - 50) * {'M1':0.15,'M3':0.20,'M6':0.25,'Y1':0.30}.get(hk,0.2))
            prob    = max(5, min(95, pct + cot_adj))
            apx     = round(px_now * (1 + av/100), 1 if px_now < 1000 else 0)
            bpx     = round(px_now * (1 + mn/100), 1 if px_now < 1000 else 0)
            bupx    = round(px_now * (1 + mx/100), 1 if px_now < 1000 else 0)
            col     = P['teal'] if prob>=65 else P['green'] if prob>=50 else P['amber'] if prob>=40 else P['red']

            proj_cards.append(html.Div([
                html.Div(style={'height':'3px','background':col,'borderRadius':'3px 3px 0 0','margin':'-12px -14px 10px'}),
                html.Div(hl,  style={'fontSize':'10px','fontWeight':'600','color':P['sub'],'textTransform':'uppercase','marginBottom':'2px'}),
                html.Div(hd,  style={'fontSize':'10px','color':P['muted'],'marginBottom':'8px'}),
                html.Div([
                    html.Span(f'{apx}', style={'fontSize':'21px','fontWeight':'600','color':col}),
                    html.Span(f' {unit}', style={'fontSize':'11px','color':P['sub']}),
                ], style={'marginBottom':'2px'}),
                html.Div(f'Bear: {bpx}  ·  Bull: {bupx}', style={'fontSize':'10px','color':P['muted'],'marginBottom':'8px'}),
                html.Div(f'Bull probability: {prob}%', style={'fontSize':'11px','fontWeight':'600','color':col,'marginBottom':'4px'}),
                html.Div(style={'height':'6px','background':P['border'],'borderRadius':'3px','overflow':'hidden','marginBottom':'4px'}, children=[
                    html.Div(style={'height':'100%','width':f'{prob}%','background':col,'borderRadius':'3px'})
                ]),
                html.Div(f'{nb}/{len(rets)} analogs bull · avg {av:+.1f}%', style={'fontSize':'10px','color':P['muted']}),
            ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'8px','padding':'12px 14px','flex':'1','minWidth':'130px'}))

        # Signal panel
        gs_col = P['red'] if gs>=80 else P['amber'] if gs>=60 else P['muted']
        ns_col = P['red'] if net_s>=80 else P['amber'] if net_s>=60 else P['muted']
        gl_col = P['teal'] if gl_asc<=20 else P['green'] if gl_asc<=40 else P['muted']

        signal_panel = html.Div([
            html.Div(style={'display':'flex','justifyContent':'space-between','alignItems':'center','marginBottom':'12px'}, children=[
                html.Div([
                    html.Span(comm, style={'fontSize':'18px','fontWeight':'600','color':P['text']}),
                    html.Span(f'  {sector}  ·  {meta["contract"]}  ·  {px_now} {unit}',
                              style={'fontSize':'12px','color':P['sub'],'marginLeft':'8px'}),
                ]),
                html.Div([
                    html.Span(f'{cot_icon} {cot_signal}', style={
                        'fontSize':'14px','fontWeight':'600','color':'#fff',
                        'background':cot_col,'padding':'5px 14px','borderRadius':'5px',
                    }),
                    html.Div(f'COT bull score: {bull_score}/100',
                             style={'fontSize':'10px','color':P['sub'],'marginTop':'3px','textAlign':'center'}),
                ]),
            ]),
            html.Div(style={'display':'flex','gap':'8px','flexWrap':'wrap','marginBottom':'10px'}, children=[
                html.Div([
                    html.Div(lbl, style={'fontSize':'10px','color':P['sub'],'marginBottom':'2px'}),
                    html.Div(f'{val:.0f}th', style={'fontSize':'17px','fontWeight':'600','color':col,'lineHeight':'1'}),
                    html.Div('pctile', style={'fontSize':'10px','color':P['sub']}),
                ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'6px',
                          'padding':'8px 12px','textAlign':'center','flex':'1','minWidth':'80px'})
                for lbl, val, col in [
                    ('Gross Short',    gs,          gs_col),
                    ('Net Short',      net_s,       ns_col),
                    ('Gross Long ↑low=bull', gl_asc, gl_col),
                    ('Short %OI',      short_oi,    P['red'] if short_oi>=80 else P['muted']),
                    ('Net %OI',        net_oi,      P['red'] if net_oi>=80 else P['muted']),
                    ('Long %OI ↑low=bull', long_oi_asc, P['teal'] if long_oi_asc<=20 else P['muted']),
                ]
            ]),
            html.Div(style={'display':'flex','gap':'8px','flexWrap':'wrap'}, children=[
                html.Div([
                    html.Div('MM net', style={'fontSize':'10px','color':P['sub'],'marginBottom':'2px'}),
                    html.Div(f"{bcom.get('mm_net',0):+,.0f}",
                             style={'fontSize':'13px','fontWeight':'600',
                                    'color':P['red'] if bcom.get('mm_net',0)<0 else P['teal']}),
                    html.Div('contracts', style={'fontSize':'10px','color':P['sub']}),
                ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'6px','padding':'8px 12px','flex':'1'}),
                html.Div([
                    html.Div('Producer net', style={'fontSize':'10px','color':P['sub'],'marginBottom':'2px'}),
                    html.Div(f"{bcom.get('prod_net',0):+,.0f}",
                             style={'fontSize':'13px','fontWeight':'600',
                                    'color':P['green'] if bcom.get('prod_net',0)>0 else P['amber']}),
                    html.Div('contracts', style={'fontSize':'10px','color':P['sub']}),
                ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'6px','padding':'8px 12px','flex':'1'}),
                html.Div([
                    html.Div('Swap Dealer net', style={'fontSize':'10px','color':P['sub'],'marginBottom':'2px'}),
                    html.Div(f"{bcom.get('swap_net',0):+,.0f}",
                             style={'fontSize':'13px','fontWeight':'600','color':P['blue']}),
                    html.Div('contracts', style={'fontSize':'10px','color':P['sub']}),
                ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'6px','padding':'8px 12px','flex':'1'}),
                html.Div([
                    html.Div('Other Rept net', style={'fontSize':'10px','color':P['sub'],'marginBottom':'2px'}),
                    html.Div(f"{bcom.get('other_net',0):+,.0f}",
                             style={'fontSize':'13px','fontWeight':'600',
                                    'color':P['teal'] if bcom.get('other_net',0)>0 else P['red']}),
                    html.Div('contracts', style={'fontSize':'10px','color':P['sub']}),
                ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'6px','padding':'8px 12px','flex':'1'}),
            ]),
        ], style={'background':P['panel'],'border':f'1.5px solid {cot_col}','borderRadius':'10px','padding':'14px 16px','marginBottom':'14px'})

        # Analog table
        acols = [P['red'],P['amber'],P['purple'],P['teal'],P['coral']]
        def ret_cell(v):
            if v is None: return html.Td('—', style={'padding':'5px 8px','color':P['muted'],'textAlign':'right'})
            c = P['teal'] if v>3 else P['green'] if v>0 else P['amber'] if v>-5 else P['red']
            return html.Td(f'{v:+.1f}%', style={'padding':'5px 8px','color':c,'fontWeight':'600','textAlign':'right','fontSize':'11px'})

        th_style = {'padding':'5px 8px','color':P['sub'],'fontSize':'10px','borderBottom':f'1px solid {P["border"]}'}
        analog_tbl = html.Div([
            html.Div('Closest analog years — by COT positioning',
                     style={'fontSize':'11px','fontWeight':'600','color':P['sub'],'marginBottom':'6px'}),
            html.Table([
                html.Tr([html.Th('Analog',th_style),html.Th('Entry',{**th_style,'textAlign':'right'}),
                         html.Th('Match',{**th_style,'textAlign':'right'}),
                         html.Th('+1M',{**th_style,'textAlign':'right'}),html.Th('+3M',{**th_style,'textAlign':'right'}),
                         html.Th('+6M',{**th_style,'textAlign':'right'}),html.Th('+1Y',{**th_style,'textAlign':'right'})]),
            ] + [
                html.Tr([
                    html.Td(a['yr'], style={'padding':'5px 8px','color':acols[i%5],'fontWeight':'600','fontSize':'11px'}),
                    html.Td(f"{a['entry']} {unit}", style={'padding':'5px 8px','color':P['sub'],'fontSize':'11px','textAlign':'right'}),
                    html.Td(f"{a['match']}%", style={'padding':'5px 8px','color':P['blue'],'fontSize':'11px','textAlign':'right'}),
                    ret_cell(a['fwd'].get('M1')), ret_cell(a['fwd'].get('M3')),
                    ret_cell(a['fwd'].get('M6')), ret_cell(a['fwd'].get('Y1')),
                ]) for i,a in enumerate(analogs)
            ], style={'width':'100%','borderCollapse':'collapse','fontSize':'11px'}),
        ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'8px','padding':'12px 14px','marginBottom':'14px'})

        # Synthesis verdict
        def avg_ret(hk):
            rets = [a['fwd'][hk] for a in analogs if hk in a['fwd']]
            return float(np.mean(rets)) if rets else 0
        def bull_pct(hk):
            rets = [a['fwd'][hk] for a in analogs if hk in a['fwd']]
            return round(sum(1 for r in rets if r>0)/len(rets)*100) if rets else 50
        def dir_str(hk):
            av = avg_ret(hk)
            return ('bullish ↑', P['teal']) if av>3 else ('bearish ↓', P['red']) if av<-3 else ('neutral →', P['amber'])

        d1,c1 = dir_str('M1'); d6,c6 = dir_str('M6'); dy,cy = dir_str('Y1')

        verdict = html.Div([
            html.Div(f'Synthesis — {comm}  ·  {cot_icon} {cot_signal}  (bull score {bull_score}/100)',
                     style={'fontSize':'12px','fontWeight':'600','color':P['text'],'marginBottom':'8px'}),
            html.Div([html.Span('1–3M: ',style={'fontWeight':'600','color':P['text']}),
                      html.Span(d1,style={'color':c1,'fontWeight':'600'}),
                      html.Span(f' — {bull_pct("M1")}% of analogs positive, avg {avg_ret("M1"):+.1f}%. ')],
                     style={'marginBottom':'4px','fontSize':'11px','lineHeight':'1.7'}),
            html.Div([html.Span('6M: ',style={'fontWeight':'600','color':P['text']}),
                      html.Span(d6,style={'color':c6,'fontWeight':'600'}),
                      html.Span(f' — {bull_pct("M6")}% of analogs positive, avg {avg_ret("M6"):+.1f}%. ')],
                     style={'marginBottom':'4px','fontSize':'11px','lineHeight':'1.7'}),
            html.Div([html.Span('1Y: ',style={'fontWeight':'600','color':P['text']}),
                      html.Span(dy,style={'color':cy,'fontWeight':'600'}),
                      html.Span(f' — {bull_pct("Y1")}% of analogs positive, avg {avg_ret("Y1"):+.1f}%. '),
                      *([] if net_s<85 else [html.Span(
                          f'MM at {net_s:.0f}th net short pctile — historical 1Y base rate: avg +17%, 72% bull.',
                          style={'color':P['blue']})])],
                     style={'fontSize':'11px','lineHeight':'1.7'}),
        ], style={'background':P['panel'],'border':f'0.5px solid {P["border"]}','borderRadius':'8px','padding':'14px 16px','marginBottom':'12px'})

        footer = html.Div(
            f'Data: CFTC Disaggregated COT | Bloomberg BCOM COT Analyzer | USDA. '
            f'Prices in {unit}. COT bull score = 35%×gross_short + 35%×net_short + 20%×(100-gross_long) + 10%×(100-long_%OI). '
            f'Not trading advice.',
            style={'fontSize':'10px','color':P['muted'],'lineHeight':'1.6',
                   'borderTop':f'0.5px solid {P["border"]}','paddingTop':'8px'})

        return html.Div([
            signal_panel,
            html.Div('Price projections — multi-factor model (COT + historical analogs)',
                     style={'fontSize':'11px','fontWeight':'600','color':P['sub'],
                            'marginBottom':'8px','textTransform':'uppercase','letterSpacing':'0.4px'}),
            html.Div(proj_cards, style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'14px'}),
            analog_tbl,
            verdict,
            footer,
        ])


    return html.Div()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print('  ╔══════════════════════════════════════════════════════╗')
    print('  ║  Cotton Belt Drought & COT Positioning Dashboard     ║')
    print(f'  ║  Version: {__version__:<45}║')
    print('  ╚══════════════════════════════════════════════════════╝')
    print()
    print('  Starting server...')
    print('  Open  http://127.0.0.1:8050  in your browser')
    print()
    print('  Tabs:')
    print('    1  Production & analogs')
    print('    2  Seasonal drought profile')
    print('    3  Futures & planting signal')
    print('    4  State detail')
    print('    5  COT — About & guide')
    print('    6  COT — Positioning heatmap')
    print('    7  COT — Price projections')
    print(f'  COT heatmap  : {len(HEATMAP_DATA)} commodities ({"embedded" if not _heat_loaded else "file"})')
    print(f'  COT proj     : {len(PROJECTIONS)} commodities ({"embedded" if not _proj_loaded else "file"})')
    print(f'  COT history  : {len(MULTI_COT)} commodities')
    print()
    app.run(debug=True, port=8050)
